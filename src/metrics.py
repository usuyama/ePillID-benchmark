import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

class IndicesCollection:
    def __init__(self):
        self.indices_dict = {}

    def add(self, phase, key, indices_list):
        if phase not in self.indices_dict:
            self.indices_dict[phase] = {}

        if key not in self.indices_dict[phase]:
            self.indices_dict[phase][key] = []

        self.indices_dict[phase][key].append(indices_list)

    def __getitem__(self, slice_key):
        return self.indices_dict[slice_key]


class MetricsCollection:
    def __init__(self):
        self.metrics = {}

    def add(self, phase, key, value, count):
        if phase not in self.metrics:
            self.metrics[phase] = {}

        if key not in self.metrics[phase]:
            self.metrics[phase][key] = AverageMeter(name=key)

        self.metrics[phase][key].add(value, count)

    def __getitem__(self, slice_key):
        return self.metrics[slice_key]


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def add(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(value)

    def best(self, mode='auto'):
        if len(self.history) == 0:
            return None, None

        if mode == 'auto':
            if ('acc' in self.name) or ('precision' in self.name):
                mode = 'max'
            else:
                mode = 'min'

        if mode == 'min':
            best_index = np.argmin(self.history)
            best_value = np.min(self.history)
        elif mode == 'max':
            best_index = np.argmax(self.history)
            best_value = np.max(self.history)
        else:
            raise Exception('mode not supported: ' + mode)

        return best_value, best_index


def classification_accuracy(outputs, classes, topk=(1,)):
    # outputs.shape=(#images, #classes)
    # classes.shape=(#classes)
    if outputs.shape[0] != classes.shape[0]:
        print(outputs.shape, classes.shape)
        raise

    with torch.no_grad():
        maxk = max(topk)
        batch_size = classes.size(0)

        # classes is the correct-indexes
        # get argmax-indxes of topk outputs
        _, top_predicted_indexes = outputs.topk(maxk, 1, True, True)
        top_predicted_indexes = top_predicted_indexes.t()
        correct = top_predicted_indexes.eq(classes.view(1, -1).expand_as(top_predicted_indexes))

        accuracies = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            accuracies.append(correct_k.mul_(1.0 / batch_size))

        return accuracies

def microavg_precision(predictions_df, report_k_prec_indices = 4, do_pr_plot = True ):
    with torch.no_grad():
        n_classes = len(predictions_df.score[0])
        size = len(predictions_df)

        precision_metrics = {}

        correct_indices = np.stack(predictions_df.correct_index.values, axis = 0)
        correct_indices = np.squeeze(correct_indices)

        scores = np.stack(predictions_df.score.values, axis = 0)
        labels = np.zeros((size, n_classes), dtype=np.int)
        labels[np.arange(len(labels)), correct_indices] = 1

        # print('Calculating precision. Labels {} and scores {}'.format(labels.shape, probs.shape))
        m_average_p = average_precision_score(
            labels,
            scores,
            average="micro"
            )
        # average="micro": Calculate metrics globally by considering each element of the label indicator matrix as a label.

        precision_metrics['avg-precision'] = m_average_p

        #find lowest prec indices
        average_precision = dict()
        for i in range(n_classes):
            average_precision[i] = average_precision_score(
                labels[:, i],
                scores[:, i])

        sorted_precind_list = sorted( ((v, k) for k, v in average_precision.items()), reverse=False)
        sorted_precind_list = [ (k, v) for (v, k) in sorted_precind_list ]

        if report_k_prec_indices <= 0:
            precision_metrics['sorted_indices'] = sorted_precind_list
        else:
            precision_metrics['lp_indices'] = sorted_precind_list[: report_k_prec_indices]
            precision_metrics['hp_indices'] = sorted_precind_list[-report_k_prec_indices : ]

        if do_pr_plot:
            import matplotlib
            matplotlib.use('Agg') #backend that doesn't display to the user
            import matplotlib.pyplot as plt

            #now create a PR-curve plot
            precision, recall, _ = precision_recall_curve(
                labels.ravel(),
                scores.ravel()
                )

            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(
                'Average Precision Score, All-Classes Micro-Average: AP={0:0.3f}'
                .format(m_average_p)
            )

            precision_metrics['PR-curve'] = plt


        return precision_metrics

def all_avg_precision(predictions_df, do_pr_plot=True, per_class=True):
    n_classes = len(predictions_df.score[0])
    size = len(predictions_df)

    scores = np.stack(predictions_df.similarity.values, axis=0)
    probs = np.stack(predictions_df.score.values, axis=0)
    scores_index = np.argsort(-scores, axis=1)

    correct_indices = np.stack(predictions_df.correct_index.values, axis=0)
    correct_indices = np.squeeze(correct_indices)

    labels = np.zeros((size, n_classes), dtype=np.int)
    labels[np.arange(len(labels)), correct_indices] = 1

    print("all_avg_precision", predictions_df.shape, labels.shape, scores.shape, n_classes, size)

    precision_metrics = {}

    precision_metrics['map'] = mapk(correct_indices, scores_index)
    precision_metrics['map_at_1'] = mapk(correct_indices, scores_index, k=1)
    #precision_metrics['map_at_5'] = mapk(correct_indices, scores_index, k=5)
    #precision_metrics['map_at_10'] = mapk(correct_indices, scores_index, k=10)

    precision_metrics['gap'] = global_average_precision(labels, scores)
    precision_metrics['gap_at_1'] = global_average_precision(labels, scores, k=1)
    #precision_metrics['gap_at_5'] = global_average_precision(labels, scores, k=5)
    #precision_metrics['gap_at_10'] = global_average_precision(labels, scores, k=10)

    #precision_metrics['gap-prob'] = global_average_precision(labels, probs)
    #precision_metrics['gap_at_1-prob'] = global_average_precision(labels, probs, k=1)
    #precision_metrics['gap_at_5-prob'] = global_average_precision(labels, probs, k=5)
    #precision_metrics['gap_at_10-prob'] = global_average_precision(labels, probs, k=10)

    precision_metrics['micro-ap'] = average_precision_score(labels, scores, average="micro")
    #precision_metrics['sample-ap'] = average_precision_score(labels, scores, average="samples")

    #precision_metrics['micro-ap-prob'] = average_precision_score(labels, probs, average="micro")
    #precision_metrics['sample-ap-prob'] = average_precision_score(labels, probs, average="samples")

    #assert abs(precision_metrics['map'] - precision_metrics['sample-ap']) < 0.01
    #assert abs(precision_metrics['map'] - precision_metrics['sample-ap-prob']) < 0.01

    # per-class ap
    if per_class:
        average_precision = dict()
        for i in range(n_classes):
            average_precision[i] = average_precision_score(labels[:, i], scores[:, i])

        sorted_precind_list = sorted(((v, k) for k, v in average_precision.items()), reverse=False)
        sorted_precind_list = [(k, v) for (v, k) in sorted_precind_list]
        precision_metrics['sorted_indices'] = sorted_precind_list

    if do_pr_plot:
        import matplotlib
        matplotlib.use('Agg') #backend that doesn't display to the user
        import matplotlib.pyplot as plt

        #now create a PR-curve plot
        precision, recall, _ = precision_recall_curve(labels.ravel(), scores.ravel())

        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average Precision Score, All-Classes Micro-Average: AP={0:0.3f}'
            .format(precision_metrics['micro-ap'])
        )

        precision_metrics['PR-curve'] = plt


        return precision_metrics

def global_average_precision(labels, scores, k=None):
    """
    labels: one-hot vector (# samples x # classes)
    scores: (# samples x # classes)
    """
    if k is None:
        k = len(labels[0])

    flat_labels = []
    flat_scores = []
    scores_index = np.argsort(-scores, axis=1)

    # TODO: should be a better way?
    for i in range(len(labels)):
        flat_labels += labels[i][scores_index[i][:k]].tolist()
        flat_scores += scores[i][scores_index[i][:k]].tolist()

    return average_precision_score(flat_labels, flat_scores)


def apk(actual, predicted, k=None):
    """
    Original: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list of int (label)
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list of int (label)
             A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    assert not isinstance(actual, float)
    if isinstance(actual, (int, np.int64)):
        actual = [actual]
    if isinstance(actual, np.ndarray):
        actual = actual.tolist()

    if not k:
        k = len(predicted)
    elif len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=None):
    """
    Original: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def create_prec_inds_str(prec_indices, label_encoder):
    prec_indices_string = ''
    for l, p in prec_indices:
        prec_indices_string += "( {}, {:.4f} ), ".format(label_encoder.classes_[l], p)

    return prec_indices_string


def probability_of_correct_class(preds, classes):
    '''
    get the predicted probability of the correct class
    also get the position of the correct class in the sorted probabilities
    '''
    preds = preds.softmax(1).detach().cpu().numpy()
    prob_of_correct_indexes = preds[np.arange(preds.shape[0]), classes]

    sorted_preds = np.sort(preds, axis=1)  # ascending

    positions_of_correct_indexes = []
    for b in range(sorted_preds.shape[0]):
        num_classes = preds.shape[1]
        position_of_correct = num_classes - np.searchsorted(sorted_preds[b], prob_of_correct_indexes[b])
        positions_of_correct_indexes.append(position_of_correct)

    return prob_of_correct_indexes, positions_of_correct_indexes


def target_group_averages(targets, values, device):
    with torch.no_grad():
        targets = torch.squeeze(targets).long().to(device)
        values = torch.squeeze(values).to(device)

        unique_targets = targets.unique(sorted=True)

        if len(unique_targets) > 1:
            target_value_count = torch.stack([(targets == target_val).sum() for target_val in unique_targets])
            sums = torch.zeros(len(unique_targets)).to(device)
            sums.scatter_add_(0, targets, values)
            avgs = sums/target_value_count.float()

            np_sums = sums.data.cpu().numpy()
            np_avgs = avgs.data.cpu().numpy()
            np_unique_targets = unique_targets.data.cpu().numpy()
        else:
            np_sums =  np.array([ values.sum().item() ])
            np_avgs =  np.array([ values.mean().item() ])
            np_unique_targets = np.array([ unique_targets.item() ])

        return zip(np_unique_targets, np_avgs, np_sums)


def microavg_precision_from_dists(targets, distances, do_pr_plot=False):
    with torch.no_grad():
        precision_metrics = {}

        labels = targets.data.cpu().numpy()
        scores = -distances.data.cpu().numpy()

        m_average_p = average_precision_score(
            labels,
            scores,
            average="micro"
            )
        precision_metrics['avg-precision'] = m_average_p

        if do_pr_plot:
            import matplotlib
            matplotlib.use('Agg') #backend that doesn't display to the user
            import matplotlib.pyplot as plt

            #now create a PR-curve plot
            precision, recall, _ = precision_recall_curve(
                labels.ravel(),
                scores.ravel()
                )

            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(
                'Average Precision Score, All-Classes Micro-Average: AP={0:0.3f}'
                .format(m_average_p)
            )

            precision_metrics['PR-curve'] = plt

    return precision_metrics

if __name__ == '__main__':
    n_class = 3
    n_sample = 6

    pred = np.random.rand(n_sample, n_class)
    actual = np.zeros((n_sample, n_class))
    actual[:, 0] = 1
    actual_index = [np.where(r==1)[0] for r in actual]
    pred_index = np.argsort(-pred, axis=1)
    print(pred)
    print(actual)
    print(actual_index)

    print('=' * 30)
    for i in range(n_sample):
        print(apk(actual_index[i], pred_index[i], k=1), pred_index[i])
        print(apk(actual_index[i], pred_index[i], k=2), pred_index[i])
        print(apk(actual_index[i], pred_index[i]), pred_index[i])
        print('-' * 30)

    map_at_all = mapk(actual_index, pred_index, k=n_class)
    map_at_1 = mapk(actual_index, pred_index, k=1)
    print('map')
    print(map_at_all, map_at_1)

    ap_sample = average_precision_score(actual, pred, average="samples")
    ap_micro = average_precision_score(actual, pred, average="micro")
    print('ap sample, micro')
    print(ap_sample, ap_micro)

    gap1 = global_average_precision(actual, pred, k=1)
    gap2 = global_average_precision(actual, pred, k=2)
    gap = global_average_precision(actual, pred)

    print('gap')
    print(gap1, gap2, gap)

    # map and ap_samples should be same
    assert abs(ap_sample - map_at_all) < 0.01
    assert abs(ap_micro - gap) < 0.01
