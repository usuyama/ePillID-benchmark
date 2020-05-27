import numpy as np
import torch
import warnings

from itertools import combinations

def pdist(vectors):
    l2norm = vectors.pow(2).sum(dim=1)
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + l2norm.view(1, -1) + l2norm.view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """
    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels, is_front=None, is_ref=None, distance_matrix=None, only_top_negatives=True):
        '''
        is_front should be None for the model-level front/back aggregation
        '''
        assert embeddings.shape[0] == len(labels), f"{embeddings.shape}, labels={labels}"

        if distance_matrix is None:
            if self.cpu:
                embeddings = embeddings.cpu()
            distance_matrix = pdist(embeddings)

        if type(labels) != np.ndarray:
            labels = labels.cpu().data.numpy()

        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        all_pairs_same_labels = labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]
        if is_front is not None:
            assert len(is_front) == len(labels)

            is_front = (is_front.cpu().data.numpy() == 1)
            all_pairs_same_sides = is_front[all_pairs[:, 0]] == is_front[all_pairs[:, 1]]
        else:
            all_pairs_same_sides = np.array([True] * len(all_pairs))

        if is_ref is not None:
            assert len(is_ref) == len(labels)

            is_ref = (is_ref.cpu().data.numpy() == 1)
            all_pairs_diff_refcons = np.logical_xor(is_ref[all_pairs[:, 0]], is_ref[all_pairs[:, 1]])

        else:
            all_pairs_diff_refcons = np.array([True] * len(all_pairs))

        positive_pairs = all_pairs[(all_pairs_same_labels & all_pairs_same_sides & all_pairs_diff_refcons).nonzero()]
        if len(positive_pairs) == 0:
            warnings.warn(f"No positive pairs were found. labels: {set(labels)}, is_front: {is_front}, is_ref: {is_ref}")

            positive_pairs = None

        negative_pairs = all_pairs[((~all_pairs_same_labels) & all_pairs_diff_refcons).nonzero()]
        if len(negative_pairs) == 0 :
            warnings.warn(f"No negatives pairs were found. labels: {set(labels)}, is_front: {is_front}, is_ref: {is_ref}")

            top_negative_pairs = None
        else:
            if only_top_negatives:
                if positive_pairs is None:
                    topk = 1
                else:
                    topk = min(len(negative_pairs), len(positive_pairs))
            else:
                topk = len(negative_pairs)

            negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
            negative_distances = negative_distances.cpu().data.numpy()
            top_negatives = np.argpartition(negative_distances, topk - 1)[:topk]
            top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    # return something even when there're no hard negatives for simplicity
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else np.argmax(loss_values)


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else np.argmax(loss_values)


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.pair_selector = HardNegativePairSelector(cpu=cpu)
        self.negative_selection_fn = negative_selection_fn

    def append_triplets(self, triplets, distance_matrix, anchor_positives, negative_indices):
        ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
        for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
            loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
            loss_values = loss_values.data.cpu().numpy()
            hard_negative = self.negative_selection_fn(loss_values)
            if hard_negative is not None:
                hard_negative = negative_indices[hard_negative]
                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])


        return anchor_positive

    def get_triplets(self, embeddings, labels, is_front=None, is_ref=None, distance_matrix=None):
        if distance_matrix is None:
            if self.cpu:
                embeddings = embeddings.cpu()
            distance_matrix = pdist(embeddings)
            distance_matrix = distance_matrix.cpu()

        if type(labels) != np.ndarray:
            labels = labels.cpu().data.numpy()

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, labels,
                                                                      is_front=is_front, is_ref=is_ref,
                                                                      distance_matrix=distance_matrix, only_top_negatives=False)

        if positive_pairs is None:
            warnings.warn('The positive_pairs from HardNegativePairSelector was None')
            return None

        found_posneg_pairs = False
        triplets = []
        for label in set(labels):
            target_positive_pairs = [x.cpu().data.numpy() for x in positive_pairs if labels[x[0]] == label]
            target_negative_indexes = [x[1] if labels[x[0]] == label else x[0] if labels[x[1]] == label else None for x in negative_pairs]
            target_negative_indexes = [x.item() for x in target_negative_indexes if x is not None]
            if is_ref is not None:
                target_negative_indexes = [x for x in target_negative_indexes if is_ref[x]]
                # the anchor (1st element) should be consumer
                target_positive_pairs = [x[::-1] if is_ref[x[0]] else x for x in target_positive_pairs]

            target_positive_pairs = np.asarray(target_positive_pairs)

            if len(target_positive_pairs) == 0 or len(target_negative_indexes) == 0:
                continue

            found_posneg_pairs = True
            self.append_triplets(triplets, distance_matrix, target_positive_pairs, target_negative_indexes)

        if len(triplets) == 0:
            if found_posneg_pairs:
                warnings.warn("all the triplets were filtered since they already have negative loss-values")

            return None

        return torch.LongTensor(np.array(triplets))


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


if __name__ == '__main__':
    embeddings = torch.from_numpy(np.asarray([
        [0],
        [0],
        [1],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
    ]))
    labels = torch.from_numpy(np.array([0] * 5 + [1] * 4))
    is_front = torch.from_numpy(np.array([0, 0, 0, 1, 1, 0, 0, 1, 1]))
    is_ref = torch.from_numpy(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0]))

    print(embeddings)
    print(labels)
    print(is_front)
    print(is_ref)

    def assert_include(pairs_or_triplets, p0, p1, p2=None):
        if pairs_or_triplets.shape[1] == 2:
            assert any([True for pair in pairs_or_triplets if p0 in pair and p1 in pair])
        else:
            assert any([True for triplet in pairs_or_triplets if triplet == [p0, p1, p2]])


    def assert_exclude(pairs_or_triplets, p0, p1, p2=None):
        if pairs_or_triplets.shape[1] == 2:
            assert not any([True for pair in pairs_or_triplets if p0 in pair and p1 in pair])
        else:
            assert not any([True for triplet in pairs_or_triplets if triplet == [p0, p1, p2]])

    def pprint(*indexes):
        items = []
        for x in indexes:
            items += [f"i={x.item()}", f"label={labels[x].item()}", f"front={is_front[x].item()}", f"ref={is_ref[x].item()}", " | "]

        print(*items)

    print("#" * 20 + "HardNegativePairSelector" + "#" * 20)

    pair_selector = HardNegativePairSelector()


    print("=" * 20 + "without is_front" + "=" * 20)
    positive_pairs, top_negative_pairs = pair_selector.get_pairs(embeddings, labels)

    print(len(positive_pairs), len(top_negative_pairs))

    print("-" * 10 + ' positive pairs')
    for p1, p2 in positive_pairs:
        assert labels[p1] == labels[p2]
        pprint(p1, p2)

    print("-" * 10 + ' top_negative_pairs')
    for p1, p2 in top_negative_pairs:
        assert labels[p1] != labels[p2]
        pprint(p1, p2)

    print("=" * 20 + "with is_front" + "=" * 20)
    positive_pairs, top_negative_pairs = pair_selector.get_pairs(embeddings, labels, is_front)

    print(len(positive_pairs), len(top_negative_pairs))

    print("-" * 10 + ' positive pairs')
    for p1, p2 in positive_pairs:
        assert labels[p1] == labels[p2]
        assert is_front[p1] == is_front[p2]
        pprint(p1, p2)

    assert_include(positive_pairs, 0, 1)
    assert_exclude(positive_pairs, 0, 3)

    print("-" * 10 + ' top_negative_pairs')
    for p1, p2 in top_negative_pairs:
        assert labels[p1] != labels[p2]
        pprint(p1, p2)


    print("=" * 20 + "with is_front/is_ref" + "=" * 20)
    positive_pairs, top_negative_pairs = pair_selector.get_pairs(embeddings, labels, is_front, is_ref=is_ref)

    print(len(positive_pairs), len(top_negative_pairs))

    print("-" * 10 + ' positive pairs')
    for p1, p2 in positive_pairs:
        assert labels[p1] == labels[p2]
        assert is_front[p1] == is_front[p2]
        assert is_ref[p1] != is_ref[p2]
        pprint(p1, p2)

    assert_include(positive_pairs, 0, 2)
    assert_exclude(positive_pairs, 0, 1)

    print("-" * 10 + ' top_negative_pairs')
    for p1, p2 in top_negative_pairs:
        assert labels[p1] != labels[p2]
        assert is_ref[p1] != is_ref[p2]
        pprint(p1, p2)

    print("=" * 20 + "with is_ref" + "=" * 20)
    positive_pairs, top_negative_pairs = pair_selector.get_pairs(embeddings, labels, is_front=None, is_ref=is_ref)

    print(len(positive_pairs), len(top_negative_pairs))

    print("-" * 10 + ' positive pairs')
    for p1, p2 in positive_pairs:
        assert labels[p1] == labels[p2]
        assert is_ref[p1] != is_ref[p2]
        pprint(p1, p2)

    assert_include(positive_pairs, 0, 3)
    assert_exclude(positive_pairs, 0, 1)

    print("-" * 10 + ' top_negative_pairs')
    for p1, p2 in top_negative_pairs:
        assert labels[p1] != labels[p2]
        assert is_ref[p1] != is_ref[p2]
        pprint(p1, p2)

    print("#" * 20 + "NegativeTripletSelector" + "#" * 20)

    triplet_selector = RandomNegativeTripletSelector(1.0)

    def assert_triplet(triplet, check_front=False, check_ref=False):
        a = triplet[0]
        p = triplet[1]
        n = triplet[2]

        assert labels[a] == labels[p] and labels[a] != labels[n], triplet

        if check_front:
            assert is_front[a] == is_front[p], triplet # don't care about the negative instance

        if check_ref:
            # n should be reference images
            assert (not is_ref[a]) and is_ref[p] and is_ref[n], triplet

    print("=" * 20 + "without is_front" + "=" * 20)
    triplets = triplet_selector.get_triplets(embeddings, labels)
    print(len(triplets))
    for t in triplets:
        pprint(*t)
        assert_triplet(t)

    print("=" * 20 + "with is_front" + "=" * 20)
    triplets = triplet_selector.get_triplets(embeddings, labels, is_front)
    print(len(triplets))
    for t in triplets:
        pprint(*t)
        assert_triplet(t, check_front=True)

    print("=" * 20 + "with is_ref" + "=" * 20)
    triplets = triplet_selector.get_triplets(embeddings, labels, is_front=None, is_ref=is_ref)
    print(len(triplets))
    for t in triplets:
        pprint(*t)
        assert_triplet(t, check_front=False, check_ref=True)

    print("=" * 20 + "with is_front/is_ref" + "=" * 20)
    triplets = triplet_selector.get_triplets(embeddings, labels, is_front=is_front, is_ref=is_ref)
    print(len(triplets))
    for t in triplets:
        pprint(*t)
        assert_triplet(t, check_front=False, check_ref=True)
