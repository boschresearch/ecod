
# How many Events Make an Object? Improving Single-Frame Object Detection on the 1 Mpx Dataset

Official PyTorch implementation of "How many Events Make an Object? Improving Single-Frame Object Detection on the 1 Mpx Dataset".

To be presented at the [CVPR 2023 Workshop on Event-based Vision](https://tub-rip.github.io/eventvision2023/).

The code allows the users to reproduce and extend the results reported in the study.
Please cite the paper when reporting, reproducing or extending the results (see bottom of the README for BibTex).

## Overview

This repository implements training and validation of object detection models on event camera data.
A filtering method during training is provided which increases detection performance.
A novel bounding box memory is implemented that remembers bounding boxes as long as a detected object doesn't move.
For details, please see the paper.

## Setup

### Install

Create a conda env

```bash
# setting up the env (resolving can take some time)
conda env create -f env.yml
# if you want to have the exact versions: conda env create -f env_with_versions.yml
conda activate ecod
pip install pycocotools
# need to build c++ part once
cd ./evis
./prepare_libs.sh
pip install .
cd ../
# install the actual module
pip install -e .
```

### Paths

Before running any code, there are a few paths that have to be set in `ecod/paths.py`.

`base_dir`: Directory to store experiment results etc.

`mnist_path`: Path to mnist dataset in npy format. Used to generate the toy datasets.

`random_move_mnist36_root`: Can be left empty in the beginning. Is the root dir of the generated RM-MNIST dataset.

`random_move_debug_root`: Same as above, but for the even smaller debug-RM-MNIST.

`proph_1mpx_path`: Root directory of the unzipped 1 Mpx Dataset


## Generating toy data

To generate the RM-MNIST dataset used in the paper:
`python scripts/data/rmmnist_generate.py --savedir <savedir>`

Then, change `random_move_mnist36_root` to `<savedir>`.

To generate a small debug dataset:
`python scripts/data/rmmnist_generate.py --savedir <savedir> --debug`


## Training

The script `train.py` starts a training and also an evaluation after training.
You can pass `--test` to only do the evaluation.


### debug-RM-MNIST
```
python scripts/train.py --dataset random_move_debug_od --shape_t 4 2 360 360 --seq_n_buffers 4 --max_epochs 10 \
--batch_size 64 --bb_name resnet18 --random_crop
```

### RM-MNIST
```
python scripts/train.py --dataset random_move_mnist36_od --shape_t 1 2 360 360 --seq_n_buffers 1 --max_epochs 20 \
     --batch_size 64 --bb_name resnet18 --random_crop --hidden_dims 512 --check_val_every_n_epoch 2
```

To re-do e.g., the different filtering splits on RM-MNIST
```
python scripts/train.py --dataset random_move_mnist36_od --shape_t 1 2 360 360 --seq_n_buffers 1 --max_epochs 20 \
     --batch_size 64 --bb_name resnet18 --random_crop --hidden_dims 512 --check_val_every_n_epoch 2 --bbox_suffix_train "filtered" --bbox_suffix_test "none"
```

### 1 Mpx Dataset

Note1: Observe that here, "filtered" refers to the filtering proposed in the 1 Mpx paper. It is NOT our filtering.
This is why the test data is also 'filtered' and not 'none'.

Note2: You will need a considerable amount of CPU and GPU RAM. We used around 64 GB CPU + 32 GB GPU RAM.
```
python scripts/train.py --dataset proph_1mpx --bbox_suffix_train "filtered" --bbox_suffix_test "filtered" \
    --shape_t 4 2 360 360 --seq_n_buffers 1 --max_epochs 10 \
    --batch_size 8 --bb_name resnext50_32x4d --random_crop --random_mirror \
    --limit_val_batches 0.7
```

### ConvLSTM neck
```
# for LSTM (remember, len(hidden_dims) and len(prior_boxes_aspect_ratios) has to match)
--seq_od_name lstm --hidden_dims 128 128 128
```

### Using the memory

You can use the following script to process all predicted bounding boxes with the memory.
The predicted boxes are automatically saved after running the eval at least once.
```
python scripts/eval_memory.py --expdir /path/to/exp/ --savedir .  --box_mem --val_test val
```

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this repository, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).


## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication cited above.
It will neither be maintained nor monitored in any way.


## Contacts

Feel free to contact us personally if you have questions, need help, or need explanations.
a.p.kugele@rug.nl


## Citation

If you use this work please cite
```
@inproceedings{
kugele2023howmany,
title={How many Events Make an Object? Improving Single-Frame Object Detection on the 1 Mpx Dataset},
author={Kugele, Alexander and Pfeil, Thomas and Pfeiffer, Michael and Chicca, Elisabetta},
booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
year={2023},
month={June},
}
```
