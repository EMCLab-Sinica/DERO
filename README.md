# Deep Reorganization: Retaining Residuals for TinyML
This repository contains the training scripts for the evaluation for DERO.

All models have been trained on 8x RTX2080Ti GPUs.

## Requriements
- Python>=3.7.0 

- Pytorch>=1.7.1

## Initial steps
Clone repo and install requirements.txt in a Python>=3.7.0 environment.
```bash
pip install -r requirements.txt
```

## Training

Training for baseline models:
```bash
torchrun --nproc_per_node=8 train.py --model resnet34 --data-path <PATH_TO_DATASET> --amp --output-dir <PATH_TO_MODEL_OUTPUT> -b 64 --wd 0.00004 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0
                                             resnet50
                                             mcunet_v4
                                             densenet121
```

Training for DERO models:
```bash
torchrun --nproc_per_node=8 train.py --model resnet34dero --data-path <PATH_TO_DATASET> --amp --output-dir <PATH_TO_MODEL_OUTPUT> -b 64 --wd 0.00004 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0
                                             resnet50dero
                                             mcunet_dero_v4
                                             densenet121_dero
```

## Evaluate
Testing for models:
```bash
python train.py --model <MODEL> --data-path <PATH_TO_DATASET> -b 64 --test-only --weights <PATH_TO_MODEL>
```
