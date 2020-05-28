# ePillID Benchmark

This is the official repo for ther paper "ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification" (CVPR 2020 VL3).

```
@inproceedings{usuyama2020epillid,
  title={ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification},
  author={Usuyama, Naoto and Delgado, Natalia Larios and Hall, Amanda K and Lundin, Jessica},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020},
  url={TBD}
}
```

This repository contains data and codes for ePillID - a benchmark for developing and evaluating computer vision models for pill identification. The ePillID benchmark is designed as a low-shot fine-grained benchmark, reflecting real-world challenges for developing image-based pill identification systems.

The characteristics of the ePillID benchmark include:
* Reference and consumer images:  The reference images are taken with controlled lighting and backgrounds, and with professional equipment. The consumer images are taken with real-world settings including different lighting, backgrounds, and equipment. For most of the pills, one image per side (two images per pill type) is available from the NIH Pillbox dataset. 

![Examples of reference and consumer images in ePillID](imgs/ePillID_reference_consumer_images.png)

* Low-shot and fine-grained setting: 13k images representing 8184 appearance classes (two sides for 4092 pill types). For most of the appearance classes, there exists only one reference image, making it a challenging low-shot recognition setting.

![ePillID data distribution](imgs/ePillID_stats.png)

* The best performing model at the time of publication is a multi-head metric learning approach. Following a CNN encoder, a biliner transformation layer is used to represent fine-grained embeddings.

![Model overview](imgs/model_overview.png)

## Data

Download and unzip [ePillID_data.zip](https://pillidwus2.blob.core.windows.net/data/ePillID_data.zip?st=2020-04-30T09%3A34%3A02Z&se=2031-05-01T02%3A38%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=Xvmr0RFWOeOSw74iyQso%2F7WQHs6lTxbtyQ9jrBTuLKI%3D)

## Docker

Docker image https://hub.docker.com/r/naotous/pytorch-image

`sudo nvidia-docker pull naotous/pytorch-image:py36torch041-legacy`

## Train and evaluate models

`sudo nvidia-docker run --ipc=host -it -v $HOME:$HOME naotous/pytorch-image:py36torch041-legacy /bin/bash`

use -v mount option to access folders from Docker containers

`source activate py36`

`python train_cv.py --appearance_network resnet18 --max_epochs 10 --data_root_dir /home/naotous/ePillID_data`

NOTE: max_epochs 10 for quick testing

## Train and evaluate models on Azure ML

`python azureml_submit_runs.py ePillID_0430`

## Notes

The ePillID dataset includes customized data from two public NIH NLM databases. Please refer to their websites for original data:
* [NIH NLM Pill Image Recognition Challenge dataset](https://pir.nlm.nih.gov/challenge/)
* [NIH NLM Pillbox](https://pillbox.nlm.nih.gov/statistics.html)


For bilinear pooling implementations, we forked [fast-MPN-Cov GitHub repo](https://github.com/jiangtaoxie/fast-MPN-COV).

Please consider citing these work as well when you find our work useful in your research.

## Disclaimer

This dataset and software is released for research purpose only.
