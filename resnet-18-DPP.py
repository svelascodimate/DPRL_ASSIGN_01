"""## Package Definition"""

import numpy as np
import time
import argparse
import os.path
import torch
from torch.autograd import Variable
import json
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP

from models.resnet import ResNet18

"""## Data Loading"""

def _load_data(DATA_PATH, batch_size, world_size, rank):
     # Preparing the training data
    transforms_train = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)

    # torch.distributed's own data loader method which loads the data such that they are non-overlapping and
    # exclusive to each process
    train_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=training_set,
                                                                         num_replicas=world_size, rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True,
                                              sampler=train_data_sampler)

    # Preparing the testing data
    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    testing_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

    test_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testing_set,
                                                                        num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(dataset=testing_set, batch_size = batch_size,
                                             shuffle = False, num_workers=4, pin_memory=True,
                                             sampler=test_data_sampler)
    
    return train_loader, test_loader

def _compute_counts(y_pred, y_batch, mode='train'):
    return (y_pred==y_batch).sum().item()

def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    """initial LR decayed by 1/10 every args.lr epochs"""
    lr = learning_rate
    if (epoch > 5):
        lr = 0.001
    if (epoch >= 10):
        lr = 0.0001
    if (epoch > 20):
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

"""## Distributed Data Parallel"""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

"""## Main """

def train_model(rank, args):
    ## Defining classes for CNN classifier
    ciphar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    wandb.init(project='DPRL_ASSIGN_01', 
            config={
                   "epochs": args.num_epoches,
                   "batch_size": args.batch_size,
                   "lr": args.learning_rate,
                   "decay": args.decay, 
                   "world_size":args.world_size
                   }, name='resnet-18-DPP')

    # DPP initialization
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
  
    ##-------------------------------------------------------
    ## Step 1: Data loader to load CIPHAR data
    ##-------------------------------------------------------
    DATA_PATH = "./data/"
    train_loader, test_loader=_load_data(DATA_PATH, args.batch_size, args.world_size, rank)

    ##-------------------------------------------------------
    ## Step 2: load the MLP model in model.py file using DPP
    ##-------------------------------------------------------
    model = ResNet18(len(ciphar_classes)).to(rank)
    model = DDP(model, device_ids=[rank])
    
    ## --------------------------------------------------
    ## Step 3: define the LOSS FUNCTION: cross-entropy
    ## --------------------------------------------------
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)  ## optimizer
    loss_fun = nn.CrossEntropyLoss().to(rank)    ## cross entropy loss
    
    ##  model training
    iteration = 0
    if True:
        for epoch in range(args.num_epoches): #10-50

          model = model.train() ## model training

          ## ---------------------------------------
          ## load checkpoint below
          ## ---------------------------------------
          #_load_checkpoint(ckp_path, model, epoch, optimizer)
        
          ## learning rate
          adjust_learning_rate(args.learning_rate, optimizer, epoch, args.decay)
          for batch_id, (x_batch,y_labels) in enumerate(train_loader):
              iteration += 1
              x_batch,y_labels = x_batch.to(rank), y_labels.to(rank)
                
              ## feed input data x into model
              output_y = model(x_batch)
              ##--------------------------------------------------------------
              ## Step 4: compute loss between ground truth and predicted result
              ##---------------------------------------------------------------
              loss = loss_fun(output_y, y_labels)
                
              ##----------------------------------------------
              ## Step 5: write back propagation steps below
              ##----------------------------------------------
              optimizer.zero_grad()
              loss.backward()
              optimizer.step() # update params
                
              ##---------------------------------------------------------
              ## Step 6: get the predict result and then compute accuracy
              ##---------------------------------------------------------
              y_pred = torch.argmax(output_y.data, 1)
              accy = _compute_counts(y_pred, y_labels)/args.batch_size

              ##----------------------------------------------------------
              ## Step 7: print loss values [I have done it]
              ##----------------------------------------------------------
              if iteration%10==0:
                print('iter: {} loss: {}, accy: {}'.format(iteration, loss.item(), accy))
                wandb.log({'epoch': epoch,'iteration': iteration, 'loss':loss.item(), 'accuracy':accy})

    ##------------------------------------
    ##    model testing code below
    ##------------------------------------
    total = 0
    accy_count = 0
    confusion_matrix = np.zeros([len(ciphar_classes),len(ciphar_classes)], int)
    model.eval()
    with torch.no_grad():
        for batch_id, (x_batch,y_labels) in enumerate(test_loader):
            x_batch, y_labels = Variable(x_batch).to(rank), Variable(y_labels).to(rank)
            
            ##---------------------------------------
            ## Step 8: write the predict result below
            ##---------------------------------------
            output_y = model(x_batch)
            y_pred = torch.argmax(output_y.data, 1)

            ##--------------------------------------------------
            ## Step 9: computing the test accuracy
            ##---------------------------------------------------
            total += len(y_labels)
            accy_count += _compute_counts(y_pred, y_labels)

            ##--------------------------------------------------
            ## Step 10: Wandb Confussion Matrix
            ##---------------------------------------------------
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_labels.cpu().numpy(), preds=y_pred.cpu().numpy(),
                        class_names=ciphar_classes)})
            
            ##--------------------------------------------------
            ## Step 11: Confussion Matrix
            ##---------------------------------------------------
            for y_label_idx, y_label in enumerate(y_labels):
              confusion_matrix[y_label.item(), y_pred[y_label_idx].item()] += 1
            
    # this condition ensures that processes do not trample each other and corrupt the files by overwriting
    if rank == 0:
        testAccuracy = accy_count / total
        print("Loss: {}, Testing Accuracy: {}".format(loss.item(), testAccuracy))
        
        # Saving the model
        state = {'model': model.state_dict(), 'test_accuracy': testAccuracy, 'num_epochs' : args.num_epoches}
        if not os.path.exists('./trained'):
            os.mkdir('./trained')
        torch.save(state, './trained/ciphar10-resnet.pth')

        metric_table = wandb.Table(columns=['label', 'accuracy', 'precision', 'recall', 'f1_score'])
        label_accuracies = np.zeros(len(ciphar_classes))
        print('{0:10s} - {1}'.format('Category','Accuracy'))
        for i, r in enumerate(confusion_matrix):
            label_accuracies[i] = r[i]/np.sum(r)
            print('{0:10s} - {1:.1f}'.format(ciphar_classes[i], label_accuracies[i]))

        for label_idx, label in enumerate(ciphar_classes):
            true_positives = confusion_matrix[label_idx, label_idx]
            precision =  true_positives/(np.sum(confusion_matrix, axis=0)[label_idx])
            recall = true_positives/(np.sum(confusion_matrix, axis=1)[label_idx])
            f1_score = (2*precision*recall)/(precision+recall)
            metric_table.add_data(label,label_accuracies[label_idx], precision, recall, f1_score)
    
        wandb.log({"metrics": metric_table})  

    cleanup()

"""## Run the Model"""

def run_train_model(training_func, world_size):

  ## initialize Default hyper-parameters
  num_epoches = 150
  decay = 0.01
  learning_rate = 0.001
  batch_size = 50

  parser = argparse.ArgumentParser("PyTorch - Training RESNET on CIFAR10 Dataset")
  parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
  parser.add_argument('--learning_rate', default=learning_rate, type=float, help='Default Learning Rate')
  parser.add_argument('--batch_size', type=int, default=batch_size, help='size of the batches')
  parser.add_argument('--num_epoches', type=int, default=num_epoches, help='Total number of epochs for training')
  parser.add_argument('--decay', type=int, default=decay, help='Total number of decay for training')
  args = parser.parse_args()

  ## Wandb initialization
  wandb.login()

  mp.spawn(training_func, 
           args=(args,), 
           nprocs=world_size, 
           join=True)

if __name__ == "__main__":
  # since this example shows a single process per GPU, the number of processes is simply replaced with the
  # number of GPUs available for training.
  n_gpus = torch.cuda.device_count()
  run_train_model(train_model, n_gpus)