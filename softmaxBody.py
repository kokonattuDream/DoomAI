
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxBody(nn.Module):
    """
    Body part of the AI
    
    Recieved the signal from brain and play the action
    
    """

    def __init__(self, T):
        """
        Initialize the SoftMaxBody
        
        @param T: temperature parameter for softmax function
        """

        super(SoftmaxBody, self).__init__()
        self.T = T
        
    def forward(self, outputs):
        """
        Forward the signal from output layer of the brain to the body
        
        @param outputs: output signal of the brain
        @return actions to play
        """
        #Apply softmax function
        probs = F.softmax(outputs * self.T)
        #The final acton to play by sampling multinomial distribution of probability 
        actions = probs.multinomial()

        return actions