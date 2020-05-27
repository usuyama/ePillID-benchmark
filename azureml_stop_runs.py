from azureml.core import Workspace, Experiment
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('experiment')
parser.add_argument('--workspace-config', default="azureml_config.json")

args = parser.parse_args()
print(args)


def stop_run(r):
    status = r.get_status()
    print(f"Stopping {r.type}, {r.id}, {status}")

    if status == 'Running':
        if 'cancel' in dir(r):
            r.cancel()
        else:
            r.complete()

    for c in r.get_children():
        stop_run(c)


ws = Workspace.from_config(path=args.workspace_config)
print('=' * 40)
print(ws)

exp = Experiment(ws, args.experiment)
for run in exp.get_runs():
    stop_run(run)
