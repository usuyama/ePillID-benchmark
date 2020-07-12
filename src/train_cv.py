import arguments

args = arguments.cv_parser().parse_args()

if args.load_config is not None:
    arguments.load_config(args)

print(args)

import os
from glob import glob
import datetime
import train_nocv
import pandas as pd
import numpy as np
from metrics import all_avg_precision
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s#L%(lineno)-3s -  %(message)s",
    datefmt="%m/%d %H:%M:%S"
 )
logging.getLogger().setLevel(logging.INFO) # needed for AML
logger = logging.getLogger(__name__)

# Azure ML
from azureml.core.run import Run
run = Run.get_context()

# log arguments
for k, v in vars(args).items():
    run.tag(k, str(v))

# TODO: fix path issue when data_root_dir is not aboslute path
assert os.path.isabs(args.data_root_dir)
# find csv files and prepare args for each run
args.folds_csv_dir = os.path.join(args.data_root_dir, args.folds_csv_dir)
args.label_encoder = os.path.join(args.folds_csv_dir, "label_encoder_pytorch131.pickle")

# make sure csv_files are sorted
csv_files = glob(os.path.join(args.folds_csv_dir, "*.csv"))
args.all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
csv_files = sorted([x for x in csv_files if not x.endswith("all.csv")])

args.test_imgs_csv = csv_files.pop(-1)  # use the last fold as hold out

print("val csv files: ",csv_files)

metrics_dfs_list = []
predictions_dfs_list = []
for i, val_csv in enumerate(csv_files):
    args.val_imgs_csv = val_csv

    metrics_df, predictions_df = train_nocv.run(args)
    metrics_dfs_list.append(metrics_df)
    predictions_dfs_list.append(predictions_df)

all_metrics_df = pd.concat(metrics_dfs_list, ignore_index = True)

# files in outputs folder will be uploaded to AML outputs tab
os.makedirs('outputs', exist_ok=True)
all_metrics_df.to_csv(os.path.join('outputs', 'metrics.csv'))


all_predictions_df = pd.concat(predictions_dfs_list, ignore_index = True)

holdout_predictions_df = all_predictions_df[all_predictions_df.dataset == 'holdout'].copy()

precision_metrics = all_avg_precision(all_predictions_df)

plt = precision_metrics['PR-curve']
run.log_image(name='cv_{}_PR-curve'.format(
        datetime.datetime.now().strftime("%H%M")
        ), plot=plt)
plt.close()

num_metrics_df = all_metrics_df[~all_metrics_df.name.str.contains('indices|curve')].copy()
num_metrics_df["value"] = pd.to_numeric(num_metrics_df["value"], errors='coerce')

agg_metrics_df = num_metrics_df.groupby(['dataset', 'name'])['value'].agg([np.mean, np.std])
agg_metrics_df.to_csv(os.path.join('outputs', 'agg_metrics.csv'))

for k, v in agg_metrics_df.iterrows():
    k = "_".join(k)

    run.log("{}_mean".format(k), v['mean'])
    run.log("{}_std".format(k), v['std'])

all_predictions_df.to_csv(os.path.join('outputs', 'all_eval_predictions.csv'))
