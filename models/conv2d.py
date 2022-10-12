import torch.nn as nn

"""## Model Definition"""

class ConvNeuralNet(nn.Module):
    ##-----------------------------------------------------------
    ## define the model architecture here
    ## CIFAR-10 image input size: 3 * 32 * 32 (three input channel)
    ##-----------------------------------------------------------
    def __init__(self, num_classes):
      super(ConvNeuralNet, self).__init__()
      ##-----------------------------------------------------------
      ## define the model architecture here
      ## CIPHAR 10 image input size batch 3 * 32 * 32 (three input channel)
      ##-----------------------------------------------------------
      # Convolutional Layers
      self.cvl = nn.Sequential(
          # CVL 1
          # img     ->  [3,32,32]
          # conv2d  ->  [64,31,31]
          # maxpool ->  [64,30,30]  
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1, stride=1), 
          nn.MaxPool2d(2,1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          # CVL 2
          # img     ->  [64,30,30]
          # conv2d  ->  [32,16,16]
          # maxpool ->  [32,14,14]
          nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding=1, stride=2),
          nn.MaxPool2d(3,1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Dropout2d(0.2),
          # CVL 3
          # img     ->  [32,14,14]
          # conv2d  ->  [64,8,8]
          # maxpool ->  [64,4,4]
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1, stride=2),
          nn.MaxPool2d(2,2),
          nn.BatchNorm2d(64),
          nn.ReLU()
      )
      # Fully Connected Layers
      self.fcl = nn.Sequential(
          nn.Linear(64*4*4, 500),
          nn.BatchNorm1d(500),
          nn.ReLU(),
          nn.Linear(500, num_classes)
      )
      # Weights initialization
      self.cvl.apply(self.weights_init)
      
    # Progresses data across layers    
    def forward(self, x):
      out = self.cvl(x)  
      out = out.reshape(out.size(0), -1)
      out = self.fcl(out) 
      return out

    #weights initialization 
    def weights_init(self, m):
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)