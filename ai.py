import numpy as np
import torch
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom




class AI:
    """
    Artificial Intelligence 
    
    """

    def __init__(self, brain, body):
        """
        Intialize the AI body
        
        @param brain: CNN
        @param body: softbody
        """

        self.brain = brain
        self.body = body
        
    def __call__(self, inputs):
        """
        Forward function that propagate the signal 
        from the beginning when the brain is getting the image to the end 
        when the AI play the action
        
        @param inputs: input images
        @return actions in numpy
        """
        
        input = Variable( torch.from_numpy ( np.array(inputs, dtype = np.float32) ) ) 

        output = self.brain(input)
        actions = self.body(output)

        #torch => numpy
        return actions.data.numpy()

