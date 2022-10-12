# Assignment_1 - Deep Representation Learning. 
Author: Sebastian Alejandro Velasco Dimate

CIPHAR10 dataset was used to train 2 models:
- Conv2d -> 3 Conv2x2 and 2 FCL. 
- ResNet -> ResNet18 architecture with basic block of 2 Conv3x3. 

Both architectures were trained usinng pytorch Distributed Data Parallel (DPP)

- conv2d-DPP.py : Conv2d architecture with DataParallel
- resnet-18-DPP.py: ResNet18 architecture with DataParallel


Wandb was used to log the total accuracy and loss metrics, as well as local acuracy, precision, recall and F1 for each CIPHAR10 class.

## Prerequeisites - Deep Representation Learning. 
- PyTorch
- NVIDIA GPU (gloo)

## Installation
- create pyenv: 
    ```python -m venv .venv```
- Launch virtual environment:   
    ```source venv/bin/activate```
- Isntall dependencies> 
    ```pip install -r requirements.txt```

## Run the models
- Execute the follwing commands:
    ```python3 conv2d-DPP.py```
    ```python3 resnet-18-DPP.py```