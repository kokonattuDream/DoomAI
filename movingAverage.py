import numpy as np


class MA:
    """
    Moving Average of n steps
    
    """
    
    def __init__(self, size):
        """
        Intializethe MA
        
        @param size: size of list of rewards
        """
        
        self.list_of_rewards = []
        self.size = size;
    
    def add(self, rewards):
        """
        Add the rewards to the list
        
        @param rewards: list of rewards
        """
        
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards += rewards
            
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
            
    def average(self):
        """
        Average of the rewards
        
        @return average of rewards
        """
        return np.mean(self.list_of_rewards)
    