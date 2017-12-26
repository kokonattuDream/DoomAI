
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class CNN(nn.Module):
"""
Convolutional Neural Network which is the brain of AI

These are 3 convolutional layers to visualize images that are passed.
Then the signal will be passed to the artifical neural network to predicct the action

CNN architecture :
    Input images => Convolution => Max Pooling => Flattening => ANN

"""
    
    def __init__(self, number_actions):
        """
        Initialize the CNN
        
        @param number_actions: number of actions from doom
        """
        
        super(CNN, self).__init__()

        # in_channels => Nubmer of channels in images. black_White => 1, color => 3
        # out_channels => # of features  you want to detect
        # kernel_size => dimensions of squares go through image
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        
        # (1,80,80) is the size of images from doom
        # Connection with input layer and hidden layer
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features = 40)
        # Connection with hidden layer and output layer
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
        
    def count_neurons(self, image_dim):
        """
        Calculate the number of neurons needed in the flattening layer in CNN (input layer of ANN)
        
        @param image_dim: Dimension of original image (1x80x80) for Doom
        @return numbers of neurons
        """
        #fake image
        x = Variable(torch.rand(1, *image_dim))
        
        #Convolution layer => max pooling => apply rectified function
        x = F.relu( F.max_pool2d( self.convolution1(x), 3, 2) )
        x = F.relu( F.max_pool2d( self.convolution2(x), 3, 2) )
        x = F.relu( F.max_pool2d( self.convolution3(x), 3, 2) )
        
        #Get the number of neurons
        return x.data.view(1, -1).size(1)
        
    def forward(self, x):
        """
        Propagate the signals in all the layers of the neural network
        
        convolution => max pooling => flattening => ANN
        
        @param x: input of CNN
        @return output neuron with Q-value
        """
        
        x = F.relu( F.max_pool2d( self.convolution1(x), 3, 2) )
        x = F.relu( F.max_pool2d( self.convolution2(x), 3, 2) )
        x = F.relu( F.max_pool2d( self.convolution3(x), 3, 2) )
        
        # flattening
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
