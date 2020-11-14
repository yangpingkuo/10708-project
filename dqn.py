#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 04:48:51 2020

"""

import gym
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical, Normal
from matplotlib import animation
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage.transform import resize
import random
import time

def simulate(env, horizon, policy, render = False):
    tot_reward = 0
    frame = env.reset()
    frame_buffer = 4 * [preprocess(frame)]
    movie_frame = []
    for t in range(horizon):
        if render:
            #env.render()
            env.render()
            movie_frame.append(env.render(mode="rgb_array"))
            time.sleep(1/24)
            
        state = torch.stack(frame_buffer[-4:]) 
        action = policy(state)
        frame, reward, done, info = env.step(action)
        frame_buffer.append(preprocess(frame))
        next_state = torch.stack(frame_buffer[-4:])
        tot_reward += reward
        if done:
            break
            
    if render:    
        env.close()
            
    return tot_reward, reward, done, t, movie_frame


def preprocess(img):
    img_gray = np.mean(img, axis=2)
    img_down = resize(img_gray,(84,84))
    img_down = np.asarray(img_down,dtype = np.int8)
    return torch.from_numpy(img_down)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, state, action, next_state, reward, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, 
                                      reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, out_size, hidden_size = 256): 
        super(DQN, self).__init__()

        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(
            in_channels=NUM_FRAMES,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0)
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0)
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0)
        self.fc1 = nn.Linear(3136, hidden_size)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # In: (4, 84, 84)  Out: (32, 20, 20)
        x = F.relu(self.conv2(x))    # In: (32, 20, 20) Out: (64, 9, 9)
        x = F.relu(self.conv3(x))    # In: (64,7,7)   Out: (64,7,7)
        x = x.view(x.size()[0], -1)  # In: (32, 13, 10) Out: (4160,)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    
def predict(state):
    q_vals = Q(state.to(device).unsqueeze(0).float() / 255).squeeze()
    if np.random.rand() < EPS:
        return np.random.randint(q_vals.shape[0])
    return q_vals.argmax().item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    
    transitions = memory.sample(BATCH_SIZE)
    batch = tuple(zip(*transitions) )

    batch_state = torch.stack(batch[0]).to(device).float() / 255
    batch_action = torch.stack(batch[1]).to(device)
    batch_next_state = torch.stack(batch[2]).to(device).float() / 255
    batch_reward = torch.stack(batch[3]).to(device)
    batch_done = torch.stack(batch[4]).to(device)

    current_Q = Q(batch_state).gather(1, batch_action)

    expected_Q = batch_reward.float()
    expected_Q[~batch_done] += GAMMA * target_Q(batch_next_state[~batch_done]).max(1)[0].detach()

    loss = F.mse_loss(current_Q, expected_Q.unsqueeze(1))
    #loss = F.smooth_l1_loss(current_Q, current_Q.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    for param in Q.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()
    return loss.detach().item()

def save_frames_as_gif(frames, path='./', filename='pong_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 200.0, frames[0].shape[0] / 200.0), dpi=144)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=0)
    anim.save(path + filename, writer='imagemagick', fps=12)
    
if __name__ == "__main__":
    NUM_FRAMES = 4
    BUFFER_SIZE = 200000#100000
    BATCH_SIZE = 64
    GAMMA = 0.99
    T_MAX = 2000
    EPISODE_MAX = 100
    TARGET_UPDATE = 2000
    EPS = 1.0
    EPS_MIN = 0.1
    EPS_LEN = 100000
    INITIAL_COLLECTION=10000
    train_hist = []
    
    env = gym.make('PongDeterministic-v4')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    Q = DQN(env.action_space.n).to(device)
    ####### load model ##########
    #Q.load_state_dict(torch.load('pong_Q'))
    
    target_Q = DQN(env.action_space.n).to(device)
    target_Q.load_state_dict(Q.state_dict())
    target_Q.eval()
    
    memory = ReplayMemory(BUFFER_SIZE)
    optimizer = optim.Adam(Q.parameters())
    global_step = 0
    for i_episode in range(EPISODE_MAX):
        tot_reward = 0
        frame = env.reset()
        frame_buffer = 4 * [preprocess(frame)]
        
        t_start = time.time()
        for t in range(T_MAX):
            global_step+=1
            state = torch.stack(frame_buffer[-4:])
            #action = np.random.randint(env.action_space.n) 
            action = predict(state)
            frame, reward, done, info = env.step(action)
            frame_buffer.append(preprocess(frame))
            next_state = torch.stack(frame_buffer[-4:])
            
            memory.push(state, torch.tensor([action]), 
                        next_state,
                        torch.tensor(reward, dtype = torch.float32),
                        torch.tensor(done))
            tot_reward += reward
            if global_step<INITIAL_COLLECTION:
                continue
            loss = optimize_model()
            if done:
                break  
            if i_episode % TARGET_UPDATE == 0:
                target_Q.load_state_dict(Q.state_dict())
        EPS = max(EPS_MIN, EPS*(EPS_LEN-global_step+INITIAL_COLLECTION)/EPS_LEN)
        
        train_hist += [tot_reward]
    
        print("Epoch:%d Global step:%d Done:%s Total Reward:%.2f Time:%d Epsilon:%.2F Elapsed Time:%.2f Buffer size:%d"%(i_episode, global_step, done, tot_reward, t, EPS, time.time() - t_start, len(memory)))
    
    ###
    
    plt.figure(figsize = (10,10))
    plt.plot(train_hist)
#    plt.plot(np.arange(0,EPISODE_MAX,10),
#             np.array(train_hist).reshape(-1, EPISODE_MAX).mean(axis = 1))
    plt.xlabel('# of Episode', fontsize = 20)
    plt.ylabel('Total Reward', fontsize = 20)
    
    ###
    _ = simulate(env, 100, predict, True)
    torch.save(Q.state_dict(), 'pong_Q')
    torch.save(target_Q.state_dict(), 'pong_Q_target')
    env = gym.make('PongDeterministic-v4')

    Q = DQN(env.action_space.n).to(device)
    target_Q = DQN(env.action_space.n).to(device)
    ####### load model ##########
    
    Q.load_state_dict(torch.load('pong_Q'))
    target_Q.load_state_dict(torch.load('pong_Q_target'))
    
    reward_tot, reward, t, done, frames = simulate(env, 500, predict, True)
    save_frames_as_gif(frames[::4])
    

