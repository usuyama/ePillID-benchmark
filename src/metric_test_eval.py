import torch
import torch.nn.functional as F
import time
import os
import sys
import copy
import pandas as pd
import numpy as np

from itertools import combinations
from collections import namedtuple
from tqdm import tqdm

from metrics import MetricsCollection, AverageMeter, classification_accuracy, probability_of_correct_class, all_avg_precision
from pillid_datasets import SiamesePillID
from sanitytest_eval import ModelEvaluator

def create_simul_query_pairids(query_labels, query_sidelbls, is_ref=None):
    """
    is_ref: only pair between refs/consumers
    """
    query_pair_idxs = []

    labels = query_labels.cpu().data.numpy()
    sidelbls = query_sidelbls.cpu().data.numpy()

    if len(labels) != len(sidelbls):
        raise Exception("Length mismatch between labels and side labels")

    if is_ref is not None:
        if len(is_ref) != len(labels):
            raise Exception("Length mismatch between labels and is_ref")
        else:
            is_ref = is_ref.cpu().data.numpy()

    all_pairs = np.array(list(combinations(range(len(labels)), 2)))
    all_pairs = torch.LongTensor(all_pairs)
    all_pairs_same_labels = labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]
    all_pairs_diff_sides = np.logical_xor(sidelbls[all_pairs[:, 0]], sidelbls[all_pairs[:, 1]])

    if is_ref is not None:
        all_pairs_same_refcons = is_ref[all_pairs[:, 0]] == is_ref[all_pairs[:, 1]]

        twoside_pairs = all_pairs[np.logical_and.reduce([all_pairs_same_labels, all_pairs_diff_sides, all_pairs_same_refcons]).nonzero()]

    else:
        twoside_pairs = all_pairs[(all_pairs_same_labels & all_pairs_diff_sides).nonzero()]

    # TODO: handle when there's no pairs i.e. twoside_pairs is empty
    pair_labels = torch.LongTensor(labels[twoside_pairs[:, 0]]) #both sides works since the label is the same

    return twoside_pairs, pair_labels

#already returns negative distances as similarities
class AnnoyPwDistance:
    def __init__(self, embeddings, return_n = 10, tree_num = 20, search_num_nodes = 50, metric="euclidean", remaining_d_scale = 1.5):

        self.annoy_idx = None
        self.reset_index(embeddings, tree_num, metric)

        #call parameter setting
        self.return_n = return_n
        self.search_num_nodes = search_num_nodes
        self.dist_scale = remaining_d_scale

    def __call__(self, q_vecs):
        q_vecs = q_vecs  if type(q_vecs) == np.ndarray else q_vecs.cpu().data.numpy()

        query_num = q_vecs.shape[0]
        pw_dists_list = []

        for i in tqdm(range(query_num), disable=None):
            q = q_vecs[i, :]
            idxs, dists = self.annoy_idx.get_nns_by_vector(q, self.return_n, search_k=self.search_num_nodes, include_distances=True)
            all_dists = np.ones(self.annoy_idx.get_n_items())*self.dist_scale*np.max(dists)
            all_dists[idxs] = dists
            pw_dists_list.append(-torch.tensor(all_dists))

        return torch.stack(pw_dists_list, dim = 0)

    def reset_index(self, embeddings, tree_num = 50, metric="euclidean"):
        from annoy import AnnoyIndex

        if self.annoy_idx is not None:
            self.annoy_idx.unload()

        self.embeddings = embeddings if type(embeddings) == np.ndarray else embeddings.cpu().data.numpy()

        since = time.time()
        self.annoy_idx = AnnoyIndex(self.embeddings.shape[1], metric=metric)
        for i in range(self.embeddings.shape[0]):
            v = embeddings[i, :]
            self.annoy_idx.add_item(i, v)
        self.annoy_idx.build(tree_num)
        time_elapsed = time.time() - since
        print('Annoy index built in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#already returns negative distances as similarities
#inspired from Kaggle Humpback 1st place
class TorchPwDistance:
    def __init__(self, embeddings):
        self.reset_index(embeddings)

    def __call__(self, q_vecs):
        m, n = q_vecs.size(0), self.t_embeddings.size(1)

        qq = torch.pow(q_vecs, 2).sum(1, keepdim=True).expand(m, n)
        ee = self.embeddings2.expand(n, m).t()
        dist =  qq + ee
        dist.addmm_(1, -2, q_vecs, self.t_embeddings)
        dist = -dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return dist

    def reset_index(self, embeddings):
        self.t_embeddings = embeddings.t()
        self.embeddings2 =  torch.pow(embeddings, 2).sum(1, keepdim=True)

class CosineSimPwDistMatrix:
    def __init__(self, embeddings):
        self.reset_index(embeddings)

    def __call__(self, q_vecs):
        q_vecs_norm = q_vecs / q_vecs.norm(dim=1)[:, None].clamp(min=1e-12)

        return torch.mm(q_vecs_norm, self.transp_norm_embeddings)

    def reset_index(self, embeddings):
        embeddings_norm = embeddings / embeddings.norm(dim=1)[:, None].clamp(min=1e-12)
        self.transp_norm_embeddings = embeddings_norm.transpose(0,1)

def create_predictions_df(query_img_paths, query_labels, qry_lbl_sim_matrix, topk, device):
    # converting to probs
    scores = qry_lbl_sim_matrix
    scores = scores.to(device)
    probs = scores.softmax(1)
    max_probs, indexes = torch.max(probs, dim=1)
    max_scores, _ = torch.max(scores, dim=1)

    print(scores.shape)
    print(indexes.shape)
    print(query_labels.shape)
    print(probs.shape)
    print(max_probs.shape)
    print(max_scores.shape)

    predictions_list = []
    predictions_list += zip(
        query_img_paths, indexes.cpu().numpy(), max_probs.cpu().numpy(),
        query_labels.data.cpu().numpy(), probs.data.cpu().numpy(),
        scores.data.cpu().numpy(), max_scores.cpu().numpy()
    )
    # NOTE: zip function only iterates up to the smallest list
    assert len(predictions_list) == qry_lbl_sim_matrix.shape[0]
    assert len(predictions_list) == len(query_img_paths)

    accuracies = classification_accuracy(scores, query_labels, topk=topk)
    metrics_results = {'top1-acc': accuracies[0].item(), 'top5-acc': accuracies[1].item()}

    np.set_printoptions(threshold=sys.maxsize)
    predictions_df = pd.DataFrame(predictions_list, columns=['img_path', 'pred_index', 'prob', 'correct_index', 'score', 'similarity', 'pred_similarity'])
    # TODO: fix column names - a bit confusing
    # score: np array, normalized scores/similarities
    # prob: scalar, max of 'score'
    # similarity: np array, scores/similarities before normalization
    # pred_similarity: scalar, max of 'similarity'

    return predictions_df, metrics_results

class MetricEmbeddingEvaluator(ModelEvaluator):
    """
    Implementation for classification based on models that map to an
    embedding space to do distance-based classification
    """

    def __init__(self, multihead_model, simul_sidepairs=True, sidepairs_agg_method='post_mean', metric_evaluator_type='euclidean'):
        super(MetricEmbeddingEvaluator, self).__init__()

        self.multihead_model = multihead_model
        # aggregate front/back after calculating the distances
        self.do_agg_distance = simul_sidepairs and 'post_' in sidepairs_agg_method
        self.sidepairs_agg_method = sidepairs_agg_method
        self.metric_evaluator_type = metric_evaluator_type


    def create_embeddings_tensor(self, dataloader, device):
        outputs_list = []
        label_list = []
        sidelbl_list = []
        img_name_list = []

        pbar = tqdm(dataloader, disable=None)
        for batch_index, batch_data in enumerate(pbar):
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            img_names = batch_data['image_name']

            if 'is_front' in batch_data:
                is_front = batch_data['is_front']
            else:
                is_front = None

            with torch.no_grad():
                model_outputs = self.multihead_model.get_embedding(inputs)

                outputs_list.append(model_outputs)
                img_name_list += img_names
                label_list.append(labels)

                if is_front is not None:
                    sidelbl_list.append(is_front)

        results = {}

        results['output_tensor'] = torch.cat(outputs_list, 0)
        results['label_tensor'] = torch.cat(label_list, 0)
        results['img_name_list'] = img_name_list

        if len(sidelbl_list) > 0:
            results['sidelbl_tensor'] = torch.cat(sidelbl_list, 0)

        return results

    def eval_model(self,
                device, eval_ref_dataloaders,
                topk=(1, 5),
                do_pr_metrics=True,
                add_single_side_eval=False):

        since = time.time()
        self.multihead_model.eval() # Set all model to evaluate mode

        predictions_list = []

        # Iterate over eval data.
        eval_dataloader = eval_ref_dataloaders['eval']
        query_results = self.create_embeddings_tensor(eval_dataloader, device)

        query_outputs = query_results['output_tensor']
        query_labels = query_results['label_tensor']
        query_img_paths = query_results['img_name_list']
        print('query_img_paths', len(query_img_paths), 'nunique', len(np.unique(query_img_paths)), query_img_paths[0])
        print('query_labels', len(query_labels), 'nunique', len(np.unique(query_labels.cpu().numpy())))

        # Iterate over reference data.
        ref_dataloader = eval_ref_dataloaders['ref']
        ref_results = self.create_embeddings_tensor(ref_dataloader, device)
        ref_outputs = ref_results['output_tensor']
        ref_labels = ref_results['label_tensor']
        print(f'Completed the eval model function. query:{query_outputs.size()}, ref:{ref_outputs.size()}')

        if self.metric_evaluator_type == 'ann':
            print("Creating approximate dist provider for metric evaluator")
            pw_dist_provider = AnnoyPwDistance(ref_outputs)
        elif self.metric_evaluator_type == 'cosine':
            print("Creating Torch cosine sim provider for metric evaluator")
            pw_dist_provider = CosineSimPwDistMatrix(ref_outputs)
        elif self.metric_evaluator_type == 'euclidean':
            print("Creating Torch dist provider for metric evaluator")
            pw_dist_provider = TorchPwDistance(ref_outputs)
        else:
            raise Exception(f'Unknown argument {self.metric_evaluator_type} for type of metric evaluator')

        label_num = ref_labels.max().long().item() + 1        
        print(f'Found {label_num} classes in the references')
        assert label_num == self.multihead_model.get_original_n_classes()

        # create pairs and aggr if simul_sidepairs
        if self.do_agg_distance:
            #creates list of tuples simulating pairs
            query_pair_idxs, query_labels = create_simul_query_pairids(query_labels, query_results['sidelbl_tensor'])
            nunique_labels = len(np.unique(query_labels.numpy()))
            query_labels = query_labels.to(device)
            query_num = len(query_pair_idxs)
            query_img_paths = [[query_img_paths[p[0]], query_img_paths[p[1]]] for p in query_pair_idxs]
            nunique_imgs = len(np.unique([item for sublist in query_img_paths for item in sublist])) # flatten
            print(f'Evaluation will be performed in {len(query_pair_idxs)} simulated 2-side pairs from {nunique_imgs} consumer images of {nunique_labels} labels.')
        else:
            query_num = query_outputs.size(0)

        #TODO include some package with scatter_min support, see if compute *_dist_2ref vectorized outside loop
        print(f"Calculate distance matrix between all the labels {label_num} and consumer-queries {query_num}")
        since_queries = time.time()

        unq_ref_labels = torch.squeeze(ref_labels).long().unique(sorted=True)

        all_metrics_results = {}

        if self.do_agg_distance:
            qf_pw_sim_matrix = pw_dist_provider(query_outputs[query_pair_idxs[:, 0], :])
            qb_pw_sim_matrix = pw_dist_provider(query_outputs[query_pair_idxs[:, 1], :])

            #ref axis aggregation
            qf_lbl_sim_matrix = torch.stack([qf_pw_sim_matrix[:, ref_labels == ref_lbl].max(dim=1)[0] for ref_lbl in unq_ref_labels ], dim =1)
            qb_lbl_sim_matrix = torch.stack([qb_pw_sim_matrix[:, ref_labels == ref_lbl].max(dim=1)[0] for ref_lbl in unq_ref_labels ], dim =1)

            if add_single_side_eval:
                # calculate single-side metrics
                f_query_img_paths = [q[0] for q in query_img_paths]
                b_query_img_paths = [q[1] for q in query_img_paths]

                # front-side
                f_predictions_df, f_metrics_results = create_predictions_df(
                    f_query_img_paths, query_labels, qf_lbl_sim_matrix, topk, device)

                f_precision_metrics = all_avg_precision(f_predictions_df, per_class=False)
                f_metrics_results.update(f_precision_metrics)
                all_metrics_results.update({'f_' + k: v for k, v in f_metrics_results.items()})

                # back-side
                b_predictions_df, b_metrics_results = create_predictions_df(
                    b_query_img_paths, query_labels, qb_lbl_sim_matrix, topk, device)

                b_precision_metrics = all_avg_precision(b_predictions_df, per_class=False)
                b_metrics_results.update(b_precision_metrics)
                all_metrics_results.update({'b_' + k: v for k, v in b_metrics_results.items()})

                # single-side (both front and back)
                s_predictions_df = pd.concat([f_predictions_df, b_predictions_df], ignore_index=True)
                s_precision_metrics = all_avg_precision(s_predictions_df, per_class=False)
                all_metrics_results.update({'s_' + k: v for k, v in s_precision_metrics.items()})

                del s_predictions_df
                del f_predictions_df
                del b_predictions_df

            import gc
            gc.collect()

            # aggregate front and back sides
            q2sides_lbl_sim_matrix = torch.stack([qf_lbl_sim_matrix, qb_lbl_sim_matrix], dim = 2)
            if 'post_mean' in self.sidepairs_agg_method:
                qry_lbl_sim_matrix = q2sides_lbl_sim_matrix.mean(dim=2).squeeze()
            elif 'post_max' in self.sidepairs_agg_method:
                qry_lbl_sim_matrix = q2sides_lbl_sim_matrix.max(dim=2)[0].squeeze()
            else:
                raise f"{self.sidepairs_agg_method} not supported"

        else:
            q_pw_sim_matrix = pw_dist_provider(query_outputs)            
            #ref axis aggregation
            qry_lbl_sim_matrix = torch.stack([q_pw_sim_matrix[:, ref_labels == ref_lbl].max(dim=1)[0] for ref_lbl in unq_ref_labels ], dim =1)

        avg_time_elapsed_qry = (time.time() - since_queries) / query_num
        print('Avg. time elapsed per metric query {:.0f}m {:.0f}s'.format(avg_time_elapsed_qry // 60, avg_time_elapsed_qry % 60))
        print('qry_lbl_sim_matrix', qry_lbl_sim_matrix.shape)

        predictions_df, metrics_results = create_predictions_df(query_img_paths, query_labels, qry_lbl_sim_matrix, topk, device)
        all_metrics_results.update(metrics_results)
        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print(all_metrics_results)

        if do_pr_metrics:
            precision_metrics = all_avg_precision(predictions_df)
            all_metrics_results.update(precision_metrics)

        return all_metrics_results, predictions_df



class LogitEvaluator(ModelEvaluator):
    """
    Implementation for classification based on models that map to an
    embedding space to do distance-based classification
    """

    def __init__(self, multihead_model, simul_sidepairs=True, sidepairs_agg_method='post_mean'):
        super(LogitEvaluator, self).__init__()

        self.multihead_model = multihead_model
        # aggregate front/back after calculating the distances
        self.do_agg_distance = simul_sidepairs and 'post_' in sidepairs_agg_method
        self.sidepairs_agg_method = sidepairs_agg_method


    def create_embeddings_tensor(self, dataloader, device):
        outputs_list = []
        label_list = []
        sidelbl_list = []
        img_name_list = []

        pbar = tqdm(dataloader, disable=None)
        for batch_index, batch_data in enumerate(pbar):
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            img_names = batch_data['image_name']        
            is_front = batch_data.get('is_front', None)

            with torch.no_grad():
                logits = self.multihead_model.get_original_logits(inputs)

                outputs_list.append(logits)
                img_name_list += img_names
                label_list.append(labels)

                if is_front is not None:
                    sidelbl_list.append(is_front)

        results = {}

        results['output_tensor'] = torch.cat(outputs_list, 0)
        results['label_tensor'] = torch.cat(label_list, 0)
        results['img_name_list'] = img_name_list

        if len(sidelbl_list) > 0:
            results['sidelbl_tensor'] = torch.cat(sidelbl_list, 0)

        return results

    def eval_model(self,
                device, eval_ref_dataloaders,
                topk=(1, 5),
                do_pr_metrics=True,
                add_single_side_eval=False):

        since = time.time()
        self.multihead_model.eval() # Set all model to evaluate mode

        predictions_list = []

        # Iterate over eval data.
        eval_dataloader = eval_ref_dataloaders['eval']
        query_results = self.create_embeddings_tensor(eval_dataloader, device)

        query_outputs = query_results['output_tensor']
        query_labels = query_results['label_tensor']
        query_img_paths = query_results['img_name_list']
        print('query_img_paths', len(query_img_paths), 'nunique', len(np.unique(query_img_paths)), query_img_paths[0])
        print('query_labels', len(query_labels), 'nunique', len(np.unique(query_labels.cpu().numpy())))

        all_metrics_results = {}        
        accuracies = classification_accuracy(query_outputs, query_labels, topk=(1, 5))
        all_metrics_results['raw-top1-acc'] = accuracies[0].item()
        all_metrics_results['raw-top5-acc'] = accuracies[1].item()

        # create pairs and aggr if simul_sidepairs
        if self.do_agg_distance:
            #creates list of tuples simulating pairs
            query_pair_idxs, query_labels = create_simul_query_pairids(query_labels, query_results['sidelbl_tensor'])
            nunique_labels = len(np.unique(query_labels.numpy()))
            query_labels = query_labels.to(device)
            query_num = len(query_pair_idxs)
            query_img_paths = [[query_img_paths[p[0]], query_img_paths[p[1]]] for p in query_pair_idxs]
            nunique_imgs = len(np.unique([item for sublist in query_img_paths for item in sublist])) # flatten
            print(f'Evaluation will be performed in {len(query_pair_idxs)} simulated 2-side pairs from {nunique_imgs} consumer images of {nunique_labels} labels.')
        else:
            query_num = query_outputs.size(0)

        label_num = query_outputs.shape[1]
        print('query_outputs.shape', query_outputs.shape)
        assert label_num == self.multihead_model.get_original_n_classes()

        #TODO include some package with scatter_min support, see if compute *_dist_2ref vectorized outside loop
        print(f"Calculate distance matrix between all the labels {label_num} and consumer-queries {query_num}")
        since_queries = time.time()

        if self.do_agg_distance:
            # no need to aggregate ref-axis (done in multihead_model)
            # (# query, # labels)
            qf_lbl_sim_matrix = query_outputs[query_pair_idxs[:, 0], :]
            qb_lbl_sim_matrix = query_outputs[query_pair_idxs[:, 1], :]

            if add_single_side_eval:
                # calculate single-side metrics
                f_query_img_paths = [q[0] for q in query_img_paths]
                b_query_img_paths = [q[1] for q in query_img_paths]

                # front-side
                f_predictions_df, f_metrics_results = create_predictions_df(
                    f_query_img_paths, query_labels, qf_lbl_sim_matrix, topk, device)

                f_precision_metrics = all_avg_precision(f_predictions_df, per_class=False)
                f_metrics_results.update(f_precision_metrics)
                all_metrics_results.update({'f_' + k: v for k, v in f_metrics_results.items()})

                # back-side
                b_predictions_df, b_metrics_results = create_predictions_df(
                    b_query_img_paths, query_labels, qb_lbl_sim_matrix, topk, device)

                b_precision_metrics = all_avg_precision(b_predictions_df, per_class=False)
                b_metrics_results.update(b_precision_metrics)
                all_metrics_results.update({'b_' + k: v for k, v in b_metrics_results.items()})

                # single-side (both front and back)
                s_predictions_df = pd.concat([f_predictions_df, b_predictions_df], ignore_index=True)
                s_precision_metrics = all_avg_precision(s_predictions_df, per_class=False)
                all_metrics_results.update({'s_' + k: v for k, v in s_precision_metrics.items()})

                del s_predictions_df
                del f_predictions_df
                del b_predictions_df

            import gc
            gc.collect()

            # aggregate front and back sides
            q2sides_lbl_sim_matrix = torch.stack([qf_lbl_sim_matrix, qb_lbl_sim_matrix], dim = 2)
            if 'post_mean' in self.sidepairs_agg_method:
                qry_lbl_sim_matrix = q2sides_lbl_sim_matrix.mean(dim=2).squeeze()
            elif 'post_max' in self.sidepairs_agg_method:
                qry_lbl_sim_matrix = q2sides_lbl_sim_matrix.max(dim=2)[0].squeeze()
            else:
                raise f"{self.sidepairs_agg_method} not supported"

        else:
            qry_lbl_sim_matrix = query_outputs

        print('qry_lbl_sim_matrix.shape', qry_lbl_sim_matrix.shape)
        avg_time_elapsed_qry = (time.time() - since_queries) / query_num
        print('Avg. time elapsed per metric query {:.0f}m {:.0f}s'.format(avg_time_elapsed_qry // 60, avg_time_elapsed_qry % 60))
        print('qry_lbl_sim_matrix', qry_lbl_sim_matrix.shape)

        predictions_df, metrics_results = create_predictions_df(query_img_paths, query_labels, qry_lbl_sim_matrix, topk, device)
        all_metrics_results.update(metrics_results)
        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        if do_pr_metrics:
            precision_metrics = all_avg_precision(predictions_df)
            all_metrics_results.update(precision_metrics)

        return all_metrics_results, predictions_df


if __name__ == '__main__':
    query_labels = torch.from_numpy(np.array([0] * 7 + [1] * 4))
    query_sidelbls = torch.from_numpy(np.array([0] * 5 + [1] * 2 + [0] * 2 + [1] * 2))
    is_ref = torch.from_numpy(np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]))

    print(query_labels)
    print(query_sidelbls)
    print(is_ref)

    twoside_pairs, pair_labels = create_simul_query_pairids(query_labels, query_sidelbls)

    print('len=', len(twoside_pairs), twoside_pairs)
    print('len=', len(pair_labels), pair_labels)

    print('-' * 20 + ' with is_ref')

    twoside_pairs, pair_labels = create_simul_query_pairids(query_labels, query_sidelbls, is_ref)

    print('len=', len(twoside_pairs), twoside_pairs)
    print(len(pair_labels), pair_labels)

    print("new is_ref")
    print(is_ref[twoside_pairs[:, 0]])

    import multihead_model
    import embedding_resnet
    emb_model = embedding_resnet.EmbeddingResNet(224, 10, 0)
    model = multihead_model.MultiheadModel(emb_model, n_classes=15)
    evalator = LogitEvaluator(model)
