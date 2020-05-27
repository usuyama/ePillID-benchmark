import os
import sys
import argparse

from azureml.core.workspace import Workspace
from azureml.core import Experiment, Datastore, RunConfiguration
from azureml.train.estimator import Estimator

parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='Azure ML experiment name')
parser.add_argument('--workspace-config', default="azureml_config.json", help='Download from the Azure ML portal')
parser.add_argument('--compute', default="nc6v3", help='Azure ML training cluster')
parser.add_argument('--max_epochs', type=int, default=300)
args = parser.parse_args()

print(args)

# load workspace configuration from the config.json file
ws = Workspace.from_config(path=args.workspace_config)
print('=' * 40)
print(ws)

# create an experiment
exp = Experiment(workspace=ws, name=args.experiment)
print('=' * 40)
print(exp)

# specify a cluster
compute_target = ws.compute_targets[args.compute]
print('=' * 40)
print(compute_target)

# Mount the blob to the training container
# NOTE: (prerequisite) unzip and upload the ePillID data to the blob
data_ds = Datastore.get(ws, datastore_name='data')

rc = RunConfiguration()
rc.environment.docker.enabled = True

# Using an image from https://hub.docker.com/r/naotous/pytorch-image
# TODO: clean up the Dockerfile
rc.environment.docker.base_image = "naotous/pytorch-image:py36torch041-legacy"
rc.environment.docker.gpu_support = True

# don't let the system build a new conda environment
rc.environment.python.user_managed_dependencies = True
# point to an existing python environment in the Docker image
rc.environment.python.interpreter_path = '/app/miniconda/envs/py36/bin/python'

# flag for the user input
SUBMIT_ALL = False


def submit(script_params):
    global SUBMIT_ALL
    input_data = None

    # the files in source_directory will be uploaded to the cluster
    est = Estimator(source_directory='src',
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='train_cv.py',
                    environment_definition=rc.environment)

    print('script_params', script_params)

    while not SUBMIT_ALL and True:
        input_data = input("Submit? [Y/y/n/s]")
        if input_data in ['Y', 'y', 'n', 's']:
            break

    if input_data == 'Y':
        SUBMIT_ALL = True

    if SUBMIT_ALL or input_data == 'y':
        run = exp.submit(est)

        print('=' * 40)
        print(run)
        print('Monitor the training progress on the portal: URL=', run.get_portal_url())
    elif input_data == 'n':
        print("aborting!")
        sys.exit(0)
    else:
        print("skip!")


# define the entry script
base_script_params = {
    '--data_root_dir': data_ds.path('ePillID_data').as_mount().as_mount(),
    '--max_epochs': args.max_epochs
}

loss_params = [
    {'--contrastive_w': 0.0, '--triplet_w': 0.0, '--arcface_w': 0.0, '--ce_w': 1.0, '--focal_w': 0.0, '--dropout': 0.5},  # plain classification (logits)
    {'--contrastive_w': 1.0, '--triplet_w': 1.0, '--arcface_w': 0.1, '--ce_w': 1.0, '--focal_w': 0.0},  # multihead metric learning
]

networks = [
    #{'--appearance_network': 'resnet18'},
    # {'--appearance_network': 'resnet34'},
    #{'--appearance_network': 'resnet50', '--train_with_side_labels': '0', '--metric_simul_sidepairs_eval': '0'},
    #{'--appearance_network': 'resnet50', '--train_with_side_labels': '0', '--metric_simul_sidepairs_eval': '1'},
    #{'--appearance_network': 'resnet50', '--train_with_side_labels': '1', '--metric_simul_sidepairs_eval': '0'},
    #{'--appearance_network': 'resnet50', '--train_with_side_labels': '1', '--metric_simul_sidepairs_eval': '1'},
    {'--appearance_network': 'resnet50'},
    {'--appearance_network': 'resnet50', '--pooling': 'CBP'},
    {'--appearance_network': 'resnet50', '--pooling': 'BCNN'},
    # {'--appearance_network': 'resnet101'},
    # {'--appearance_network': 'resnet152'},
    # {'--appearance_network': 'densenet121'},
    # {'--appearance_network': 'densenet161'},
    # {'--appearance_network': 'densenet201'},
]

for l in loss_params:
    for n in networks:
        submit({**l, **n, **base_script_params})
