import torch

import time
import os
import sys
import copy
import pandas as pd
from tqdm import tqdm
from metrics import MetricsCollection, AverageMeter, classification_accuracy, probability_of_correct_class, microavg_precision
from pillid_datasets import SingleImgPillID


def create_eval_dataloaders(img_df,
                            label_encoder, torch_transform, labelcol,
                            batch_size, rotate_aug=None):
    # Get the eval images supported
    labels_df = pd.DataFrame({labelcol: label_encoder.classes_})
    img_df = pd.merge(img_df, labels_df,
                      on=[labelcol],
                      how='inner')

    # Set up data loader
    clasif_eval_dataset = SingleImgPillID(img_df, label_encoder,
                                          train=False, transform=torch_transform, labelcol=labelcol, rotate_aug=rotate_aug)

    dataloader = torch.utils.data.DataLoader(
        clasif_eval_dataset, batch_size=batch_size,
        shuffle=False, num_workers=3, drop_last=False,
        pin_memory=True)

    return dataloader, clasif_eval_dataset

class ModelEvaluator:
    """
    Base class for all model evaluators.
    Implementation should use eval_model to return metrics and predictions
    after receiving a dataloader and a device
    """

    def __init__(self):
        pass

    def eval_model(self, device, dataloader, do_pr_metrics = False):
        raise NotImplementedError


class ScoreClassifierEvaluator(ModelEvaluator):
    """
    Implementation for regular classifiers that return probabilities or 
    confidence scores for each class, e.g. fc layer with softmax
    """

    def __init__(self, model, criterion):
        super(ScoreClassifierEvaluator, self).__init__()

        self.model = model
        self.criterion = criterion
        self.results_dir = None

    def eval_model(
        self,
        device, dataloader,
        do_pr_metrics = False, 
        topk=(1, 5)):

        since = time.time()
        self.model.eval()   # Set model to evaluate mode

        batch_metrics = MetricsCollection()
        predictions_list = []

        # Iterate over data.
        for batch_data in tqdm(dataloader, disable=None):
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            img_paths = batch_data['image_name']

            # track history if only in train
            with torch.set_grad_enabled(False):
                model_outputs = self.model(inputs)
                if type(model_outputs) is dict:
                    outputs = model_outputs['classification_outputs']
                else:
                    outputs = model_outputs
                loss = self.criterion(outputs, labels)
                probs = outputs.softmax(1)
                max_probs, indexes = torch.max(probs, dim=1)

                predictions_list += zip(
                    img_paths, indexes.cpu().numpy(), max_probs.cpu().numpy(),
                    labels.data.cpu().numpy(), probs.data.cpu().numpy()
                    )
                        

            if self.results_dir is not None:
                # debug visualization
                if 'visualize_preds' in dir(self.model):
                    prob_of_corrects, positions_of_corrects = probability_of_correct_class(outputs, labels)
                    fig_titles = ["prob-of-correct-class {:.4f}, top {} position".format(x[0], x[1]) for x in zip(prob_of_corrects, positions_of_corrects)]

                    self.model.visualize_preds(inputs.cpu(), model_outputs, save_dir=self.results_dir, file_names=[os.path.basename(x) for x in img_paths], titles=fig_titles)

            # statistics
            batch_metrics.add('eval', 'loss', loss.item(), inputs.size(0))

            accuracies = classification_accuracy(outputs, labels, topk=(1, 5))
            batch_metrics.add('eval', 'top1-acc', accuracies[0].item(), inputs.size(0))
            batch_metrics.add('eval', 'top5-acc', accuracies[1].item(), inputs.size(0))

        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        metrics_results = {}
        for key, meter in batch_metrics['eval'].items():
            metrics_results[key] = meter.avg
        print(metrics_results)

        predictions_df = pd.DataFrame(predictions_list, columns=['img_path', 'pred_index', 'prob', 'correct_index', 'score'])

        if do_pr_metrics:
            precision_metrics = microavg_precision(predictions_df)
            metrics_results.update(precision_metrics)

        return metrics_results, predictions_df
