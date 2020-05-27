import argparse
import os
import json
from distutils.util import strtobool


def common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir', default="/mydata")
    parser.add_argument('--img_dir', default="classification_data")
    parser.add_argument('--supress_warnings', action='store_true')

    parser.add_argument('--optimizer', default='adam')
    parser.add_argument("--init_lr", type=float, default=1e-4, help='initial learning rate')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout rate in the final layer')
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_factor", type=float, default=0.5, help='factor of decrease in the learning rate')
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--add_persp_aug', default='1', type=strtobool, help='switches to enhaced augmentation with perspective')

    parser.add_argument('--appearance_network', default='resnet50')
    parser.add_argument('--pooling', default='GAvP', choices=['MPNCOV', 'CBP', 'BCNN', 'GAvP'], help='pooling layer for the appearance embeddings')

    parser.add_argument("--metric_margin", type=float, default=1.0, help='margin for the contrastive loss in training')
    parser.add_argument("--metric_embedding_dim", type=int, default=2048, help='dimensionality of the embedding feture vector used for the contrastive loss learning')
    parser.add_argument('--train_with_side_labels', default='1', type=strtobool, help='train with side info i.e. front and back will be treated as different classes (req. CSVs with side labels)')
    parser.add_argument('--metric_simul_sidepairs_eval', default='1', type=strtobool, help='evals holdout and val simulating per pill side img pairs (req. CSVs with side labels)')
    parser.add_argument('--sidepairs_agg', type=str, default="post_mean", choices=['post_mean', 'post_max'], help="aggregation method for embeddings. post_*: agg. after calculating the distance")
    parser.add_argument('--metric_evaluator_type', type=str, default="cosine", choices=['euclidean', 'cosine', 'ann'],
                        help="Selects Torch-based euclidean or cosine similarity or Annoy-based approx. calculation for evaluator")

    parser.add_argument("--ce_w", default=1.0, type=float)
    parser.add_argument("--arcface_w", default=0.1, type=float)
    parser.add_argument("--contrastive_w", default=1.0, type=float)
    parser.add_argument("--triplet_w", default=1.0, type=float)
    parser.add_argument("--focal_w", default=0.0, type=float)
    parser.add_argument("--focal_gamma", type=float, default=0.0, help='gamma for focal loss')

    parser.add_argument('--load_mod')
    parser.add_argument('--results_dir', default="classification_results")

    parser.add_argument('--load_config', help='load a pre-defined config from a json file')

    return parser


def nocv_parser():
    parser = common_parser()

    parser.add_argument('--all_imgs_csv', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    parser.add_argument('--val_imgs_csv', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    parser.add_argument('--test_imgs_csv', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    parser.add_argument('--label_encoder', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder.pickle")

    return parser


def cv_parser():
    parser = common_parser()

    parser.add_argument('--folds_csv_dir', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/", help='folder that contains data-splits csv files and encoder')
    parser.add_argument('--all_img_src', default="all", help='can be all pill id images or synth_date_crop')

    return parser


def load_config(args):
    print(f"Loading the predefined config: {args.load_config}")
    print("Warning: the command arguments will be overwritten by the predefined config.")

    params = json.load(open(args.load_config, 'r', encoding='utf-8'))
    for k, v in params.items():
        setattr(args, k, v)
