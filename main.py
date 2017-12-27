#Numpy and torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
    
from cnn import CNN
from ai import AI
from softmaxBody import SoftmaxBody
from image_preprocessing import image_preprocessing
from experience_replay import experience_replay
from movingAverage import MA

#Getting Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

#Building AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody( T = 1.0 )
ai = AI(brain = cnn, body = softmax_body)

#Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)



def eligibility_trace(batch):
"""
Eligibility trace (Asynchronous n-step Q-learning) that get total reward after running n-steps

@param batch: series of inputs and targets
@return updated inputs and targets
"""
    gamma = 0.99
    inputs = []
    targets = []

    for series in batch:
        input = Variable( torch.from_numpy( np.array([series[0].state, series[-1].state], dtype = np.float32) ) )
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
            
        #Get inputs and targets ready
        #First transation state
        state = series[0].state
        #Q-value of input state of the first transation
        target = output[0].data 
        #Update q-value only for the action selected in the first step of series to cumulative reward
        target[series[0].action] = cumul_reward
        
        inputs.add(state)
        targets.append(target)
        
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)




#Train the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100

ma = MA(100)

for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
        
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    
    if avg_reward >= 1500:
        print("Congratulations, your AI wins")
        break

# Closing the Doom environment
doom_env.close()
