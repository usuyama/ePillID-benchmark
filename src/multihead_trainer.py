import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import time, os, sys, inspect, copy, datetime, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from collections import defaultdict
import gc

DEBUG = False

# Azure ML
from azureml.core.run import Run
run = Run.get_context()

def get_current_lr(optimizer):
    for g in optimizer.param_groups:
        return g['lr']

from metric_utils import HardNegativePairSelector, RandomNegativeTripletSelector

from metrics import IndicesCollection, MetricsCollection, AverageMeter, microavg_precision_from_dists, classification_accuracy
from pillid_datasets import BalancedBatchSamplerPillID, SingleImgPillID
from torchvision import transforms

res_mean = [0.485, 0.456, 0.406]
res_std = [0.229, 0.224, 0.225]

torch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(res_mean, res_std)
])

def build_logid_string(args, add_timestamp=True):
    scratch = ("-scratch" if args.from_scratch else "")
    extd = ("_" + extend_ver) if args.do_appid else ""
    n_folds = "_{}flds".format(args.cv_foldn) if args.cv_foldn > 1 else ""

    param_str = "initlr{}_dr{}_lrpatience{}_lrfactor{}_{}".format(
        args.init_lr, args.dropout, args.lr_patience,
        args.lr_factor, args.appearance_network)

    param_str = param_str + scratch + extd + n_folds

    if add_timestamp:
        param_str += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

    return param_str

from sanitytest_eval import create_eval_dataloaders

from models.multihead_model import MultiheadModel
from models.embedding_model import EmbeddingModel
from models.losses import MultiheadLoss
from metric_test_eval import MetricEmbeddingEvaluator, LogitEvaluator, create_simul_query_pairids

def create_dataloaders(ref_only_df, cons_train_df, cons_val_df,
                       label_encoder, torch_transform, labelcol,
                       batch_size, add_perspective=False,
                       n_samples_per_class=6):
    # no zero-shot option for now
    train_df = pd.concat([ref_only_df] + [cons_train_df], sort = False)
    train_df = train_df.sample(frac=1.0)
    val_df = pd.concat([ref_only_df, cons_val_df])

    # Create the loaders
    train_dataset = SingleImgPillID(train_df, label_encoder,
                                    train=True,
                                    transform=torch_transform,
                                    labelcol=labelcol,
                                    add_perspective=add_perspective)
    val_dataset = SingleImgPillID(val_df, label_encoder,
                                  train=False,
                                  transform=torch_transform,
                                  labelcol=labelcol)

    batch_samplers = {
        'train': BalancedBatchSamplerPillID(train_df, batch_size=batch_size, labelcol=labelcol),
        'val': BalancedBatchSamplerPillID(val_df, batch_size=batch_size, labelcol=labelcol)
    }

    val_dataloader, _ = create_eval_dataloaders(
        cons_val_df, label_encoder, torch_transform,
        labelcol, 24)

    ref_dataloader, _ = create_eval_dataloaders(
        ref_only_df, label_encoder, torch_transform,
        labelcol, 24)

    image_datasets = {'train': train_dataset, 'val': val_dataset}

    print("train_dataset", len(train_dataset), 'val_dataset', len(val_dataset))

    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_sampler=batch_samplers[x],
        num_workers=6,
        pin_memory=True) for x in ['train', 'val']}

    dataloaders.update({'eval':val_dataloader, 'ref':ref_dataloader}) #keys match eval_model in metric_test_eval

    return dataloaders


def hneg_train_model(model, optimizer, scheduler,
                device, dataloaders,
                results_dir,
                label_encoder,
                criterion,
                num_epochs=100,
                earlystop_patience=7,
                simul_sidepairs=False,
                train_with_side_labels=True,
                sidepairs_agg='post_mean',
                metric_evaluator_type='euclidean',
                val_evaluator='metric'
                ):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    has_waited = 0
    stop_training = False

    epoch_metrics = MetricsCollection()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        evaluator = MetricEmbeddingEvaluator(model, simul_sidepairs=simul_sidepairs, sidepairs_agg_method=sidepairs_agg, metric_evaluator_type=metric_evaluator_type)
        logit_evaluator = LogitEvaluator(model, simul_sidepairs=simul_sidepairs, sidepairs_agg_method=sidepairs_agg)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Phase: {}'.format(phase))
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            batch_metrics = MetricsCollection()
            distance_records = defaultdict(list)

            # Iterate over data.
            loader = dataloaders[phase]
            # tqdm disable=None for Azure ML (no progress-bar for non-tty)
            pbar = tqdm(loader, total=len(loader), desc="Epoch {} {}".format(epoch, phase), ncols=0, disable=None)
            for batch_index, batch_data in enumerate(pbar):
                if DEBUG and batch_index > 10:
                    break

                inputs = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)

                if DEBUG:
                    print("labels", labels)
                    print(batch_data['is_front'])
                    print(batch_data['is_ref'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    all_outputs = model(inputs, labels)
                    loss_outputs = criterion(all_outputs, labels, is_front=batch_data['is_front'], is_ref=batch_data['is_ref'])

                    if loss_outputs is None:
                        warnings.warn(f"loss_outputs is None, skip this minibtach. labels: {labels}, is_front: {batch_data.get('is_front', None)}, is_ref: {batch_data.get('is_ref', None)}")

                        continue

                    logits = all_outputs['logits']
                    if train_with_side_labels:
                        # front/back is treated as different classes
                        logits = model.shift_label_indexes(logits)
                    accuracies = classification_accuracy(logits, labels, topk=(1, 5))
                    batch_metrics.add(phase, 'acc1', accuracies[0].item(), inputs.size(0))
                    batch_metrics.add(phase, 'acc5', accuracies[1].item(), inputs.size(0))

                    for prefix in ['triplet_', 'contrastive_']:
                        for n in ['distances', 'targets']:
                            k = prefix + n
                            if k in loss_outputs:
                                distance_records[k].append(loss_outputs[k])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_outputs['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        lr = get_current_lr(optimizer)
                        batch_metrics.add(phase, 'lr', lr, inputs.size(0))

                for k in ['loss', 'metric_loss', 'ce', 'arcface', 'contrastive', 'triplet', 'focal']:
                    if k in loss_outputs:
                        batch_metrics.add(phase, k, loss_outputs[k].item(), inputs.size(0))

                for prefix in ['triplet', 'contrastive']:
                    for pn in ['pos', 'neg']:
                        k = f"{prefix}_{pn}_distances"
                        if k not in loss_outputs:
                            continue

                        batch_metrics.add(phase, k, loss_outputs[k].mean().item(), loss_outputs[k].size(0))

                pbar.set_postfix(**{k.replace("contrastive", "cont").replace("triplet", "trip").replace("distances", "dist"):
                    ("{:.1e}" if k.endswith('lr') else "{:.2f}").format(meter.avg) for k, meter in batch_metrics[phase].items()})

            # finished all batches
            # copy the average batch metrics
            for key, meter in batch_metrics[phase].items():
                epoch_metrics.add(phase, key, meter.avg, 1)
                run.log('{}_{}'.format(phase, key), meter.avg)

            #avg-dist abs diff
            for prefix in ['triplet_', 'contrastive_']:
                pos_k = f"{prefix}pos_distances"
                neg_k = f"{prefix}neg_distances"
                if pos_k not in epoch_metrics[phase] or neg_k not in epoch_metrics[phase]:
                    continue

                avg_dist_diff = epoch_metrics[phase][neg_k].history[epoch] - epoch_metrics[phase][pos_k].history[epoch]
                epoch_metrics.add(phase, prefix + 'dist_diff', avg_dist_diff, 1)
                run.log('{}_{}'.format(phase, prefix + 'dist_diff'), avg_dist_diff)

                distances = torch.cat(distance_records[prefix + 'distances'], 0)
                targets = torch.cat(distance_records[prefix + 'targets'], 0)

                precision_metrics = microavg_precision_from_dists(targets, distances)

                epoch_metrics.add(phase, prefix + 'pw-avg-precision', precision_metrics['avg-precision'], 1)
                run.log('{}_{}'.format(phase, prefix + 'pw-avg-precision'), precision_metrics['avg-precision'])

            # pandas DataFrame in evaluator has memory leak
            checkpoint = 5
            if phase == 'val' and epoch % checkpoint == 0:
                print("#### Checkpoint ###")
                eval_logit = 'logit' in val_evaluator
                if eval_logit:
                    print("Evaluating logit metrics")
                    # logit eval                
                    logit_evaluator.multihead_model = model
                    metrics_results, _ = logit_evaluator.eval_model(device, dataloaders)

                    for key, value in [(key, value) for key, value in metrics_results.items() if isinstance(value, (int, float))]:
                        epoch_metrics.add(phase, key, value, 1)
                        run.log('{}_{}_logit'.format(phase, key), value)

                    del metrics_results
                    gc.collect()

                # eval using embedding distances
                eval_metric_embedding = 'metric' in val_evaluator
                if eval_metric_embedding:
                    print("Evaluating metric metrics")
                    evaluator.siamese_model = model.embedding_model # DataParallel
                    metrics_results, _ = evaluator.eval_model(device, dataloaders)

                    for key, value in [(key, value) for key, value in metrics_results.items() if isinstance(value, (int, float))]:
                        epoch_metrics.add(phase, key, value, 1)
                        run.log('{}_{}_metric'.format(phase, key), value)

                    del metrics_results
                    gc.collect()

                best_value, best_checkpoint_index = epoch_metrics['val']['micro-ap'].best(mode='max')
                if best_checkpoint_index + 1 == len(epoch_metrics['val']['micro-ap'].history):
                    has_waited = 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"Saving the best model state dict, {best_value}, {best_checkpoint_index}")
                else:
                    if has_waited >= earlystop_patience:
                        print("** Early stop in training: {} waits **".format(has_waited))
                        stop_training = True

                    has_waited += 1

                if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(-1.0 * epoch_metrics['val']['micro-ap'].value)
                else:
                    scheduler.step()

        print()  # end of epoch
        if stop_training:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    _, best_epoch = epoch_metrics['val']['micro-ap'].best(mode='max')

    best_metrics = {'best_epoch': best_epoch}
    for k, v in epoch_metrics['val'].items():
        try:
            best_metrics[k] = v.history[best_epoch]
        except:
            pass

    # load best model weights
    model.load_state_dict(best_model_wts)
    # model = model.module # DataPrallel
    return model, best_metrics

def init_mod_dev(args, label_encoder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_encoder.classes_)
    print(f"n_classes={n_classes}")

    E_model = EmbeddingModel(network=args.appearance_network, pooling=args.pooling, dropout_p=args.dropout, cont_dims=args.metric_embedding_dim, pretrained=True)

    model = MultiheadModel(E_model, n_classes, train_with_side_labels=args.train_with_side_labels)
    print(model)

    if args.load_mod:
        model.load_state_dict(torch.load(args.load_mod))

    model.to(device)

    return model, device

def train(ref_only_df,
          cons_train_df,
          cons_val_df,
          label_encoder,
          torch_transform,
          labelcol,
          batch_size,
          _,
          args,
          n_epochs,
          results_dir=None,
          add_perspective=False
          ):
    dataloaders = create_dataloaders(
        ref_only_df,
        cons_train_df,
        cons_val_df,
        label_encoder,
        torch_transform,
        labelcol,
        batch_size,
        add_perspective=add_perspective
        )

    model, device = init_mod_dev(args, label_encoder)

    if args.optimizer == 'momentum':
        optimizer = optim.SGD(list(model.parameters()), lr=args.init_lr)
    elif args.optimizer == 'adamdelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.init_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    # Reduces the LR on plateaus
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=args.lr_factor,
                                                      patience=args.lr_patience,
                                                      verbose=True)

    if results_dir is None:
        results_dir = os.path.join(args.results_dir, build_logid_string(args))

    print("Starting multihead training")
    loss_weights = {'ce': args.ce_w, 'arcface': args.arcface_w, 'contrastive': args.contrastive_w, 'triplet': args.triplet_w, 'focal': args.focal_w}
    print("loss_weights", loss_weights)
    onlinecriterion = MultiheadLoss(len(label_encoder.classes_),
        args.metric_margin, HardNegativePairSelector(),
        args.metric_margin, RandomNegativeTripletSelector(args.metric_margin),
        use_cosine=args.metric_evaluator_type == 'cosine',
        weights=loss_weights,
        focal_gamma=args.focal_gamma,
        use_side_labels=args.train_with_side_labels)

    # switch evaluator for monitoring validation performance
    val_evaluator = 'logit'
    if loss_weights['triplet'] > 0 or loss_weights['contrastive'] > 0 or loss_weights['arcface'] > 0:
        val_evaluator = 'metric'

    print(f'Will use {val_evaluator} evaluator for validation')

    model, best_val_metrics = hneg_train_model(model, optimizer,
                             exp_lr_scheduler, device, dataloaders,
                             results_dir, label_encoder, onlinecriterion,
                             num_epochs=n_epochs, 
                             earlystop_patience=3 * (args.lr_patience + 1),
                             simul_sidepairs=args.metric_simul_sidepairs_eval,
                             sidepairs_agg=args.sidepairs_agg,
                             train_with_side_labels=args.train_with_side_labels,
                             metric_evaluator_type=args.metric_evaluator_type,
                             val_evaluator=val_evaluator)

    return model, best_val_metrics
