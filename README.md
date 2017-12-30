# DoomAI

This is an AI to pass a level in Doom.

It is based on a Deep Convolutional Q-learning model with n-Step Q-Learning

Doom environment => https://gym.openai.com/envs/#doom


Convolutional Neural Network

Input images -> Convolutional layer -> Max Pooling -> Flattening -> Aritificial Neural Network

n-Step Q-Learning (Eligibility Trace):
Run N steps and calculate the total reward. Based on the result, choose a best path
