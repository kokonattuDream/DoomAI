import numpy as np
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])



class NStepProgress:
    """
    Make AI progress on several steps
    
    """
    
    def __init__(self, env, ai, n_step):
        """
        Intialize the NStepProgress
        
        @param env : environment
        @param ai: Artificial Intelligence
        @param n_step: number of steps
        """
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
    
    def __iter__(self):
        """
        Iterate through the n steps
        
        """
        
        state = self.env.reset()
        history = deque()
        reward = 0.0
        
        while True:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            
            while len(history) > self.n_step + 1:
                history.popleft()
                
            if len(history) == self.n_step + 1:
                yield tuple(history)
                
            state = next_state
            
            if is_done:
                
                if len(history) > self.n_step + 1:
                    history.popleft()
                    
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                    
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                history.clear()
    
    def rewards_steps(self):
        """
        Return all of rewards
        
        @return reward steps
        """
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


class ReplayMemory:
    """
    Experience Replay that stores memory of past series of events
    
    -Help AI study long-term correlation
    -Helps make deep Q-learning process much better by keeping long-term memory
    
    """
    def __init__(self, n_steps, capacity = 10000):
        """
        Intialize the ReplayMemory
        
        @param n_steps: memory
        @param capacity: capacity of memory default 10000
        """
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size): 
        """
        creates an iterator that returns random batches
        
        @param batch_size: size of batches
        """
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        """
        Push new step to the memory
        
        @param samples: sample 
        """
        while samples > 0:
            entry = next(self.n_steps_iter)
            # we put 200 for the current episode
            self.buffer.append(entry) 
            samples -= 1
            
        # Accumulate no more than the capacity (10000)
        while len(self.buffer) > self.capacity: 
            self.buffer.popleft()
