#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 04:48:51 2020

"""

import gym
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import animation
import matplotlib.pyplot as plt
import cv2
from cv2 import resize as imresize
from tqdm import tqdm
import psutil
import random

### Import for testing should be removed to main after
#from dqn.environment import GymEnvironment

class Enviroment(object):
    def __init__(self,config):
        self.env = gym.make(config.ENV_NAME)
        self.random_start = config.RANDOM_START
        self.dtype = config.IMG_DTYPE
    def new_game(self):
        frame = self.env.reset()
        for _ in np.arange(random.randint(0,self.random_start)):
            frame, reward, done, info = self.env.step(0)
        return preprocess(frame,self.dtype)
    
    def step(self,action):
        frame, reward, done, info = self.env.step(action)
        return preprocess(frame,self.dtype),reward,done,info
    
    def random_step(self):
        action = self.env.action_space.sample()
        frame, reward, done, info = self.step(action)
        return frame,action,reward,done,info
    
class ReplayMemory(object):
    def __init__(self, config):
        self.capacity = config.BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE
        self.position = 0
        self.count = 0
        self.history_length = config.FRAME_STACK
        #Preallocate Memory to ensure the RAM has enough capacity
        self.images = np.empty((self.capacity,config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]),dtype = config.IMG_DTYPE)
        self.actions = np.empty(self.capacity,dtype = np.uint8)
        self.rewards = np.empty(self.capacity,dtype = np.int)
        self.done = np.empty(self.capacity,dtype = np.bool)
        self.state = np.empty((self.batch_size,config.FRAME_STACK)+config.IMAGE_SIZE,dtype = np.float32)
        self.next_state = np.empty((self.batch_size,config.FRAME_STACK)+config.IMAGE_SIZE,dtype = np.float32)
        
    def push(self, image, action, reward, done):
        """Saves a transition."""
        self.images[self.position] = image
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.done[self.position] = done
        self.count = max(self.count,self.position+1)
        self.position = (self.position+1) % self.capacity
        
    def get_state(self,index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        index = (index%self.count)
        if index >= self.history_length-1:    
            return self.images[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            indexes = self._circular_index(index + 1 - self.history_length,index+1)
            return self.images[indexes, ...]
    
    def _circular_index(self,start,end):
        return np.arange(start,end)%self.count
    
    def sample(self):
        assert self.count > self.history_length
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(0, self.count - 1)
                # if wraps over current pointer, then get new one
                if self.position in self._circular_index(index - self.history_length,index+1):
                    continue
                # if wraps over episode end, then get new one
                if self.done[self._circular_index(index - self.history_length,index)].any():
                    continue
                # otherwise use this index
                break
          
            # NB! having index first is fastest in C-order matrices
            self.state[len(indexes), ...] = self.get_state(index-1)
            self.next_state[len(indexes), ...] = self.get_state(index)
            indexes.append(index)
    
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        done = self.done[indexes]
        return (torch.from_numpy(self.state),
                torch.from_numpy(self.next_state),
                torch.from_numpy(actions),
                torch.from_numpy(rewards),
                torch.from_numpy(done))
       
    def __len__(self):
        return self.count
    
    def __getitem__(self,idx):
        return  (self.get_state(idx-1),
                self.get_state(idx),
                self.actions[idx],
                self.rewards[idx],
                self.done[idx])

class DQN(nn.Module):

    def __init__(self, config): 
        super(DQN, self).__init__()
        self.device = config.DEVICE
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(
            in_channels=config.FRAME_STACK,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=0)
        out_shape = self.get_shape(np.array(config.IMAGE_SIZE),
                              padding = np.array(self.conv1.padding),
                              kernal_size = np.array(self.conv1.kernel_size),
                              stride = np.array(self.conv1.stride),
                              dilation = np.array(self.conv1.dilation))
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=0)
        out_shape = self.get_shape(out_shape,
                              padding = np.array(self.conv2.padding),
                              kernal_size = np.array(self.conv2.kernel_size),
                              stride = np.array(self.conv2.stride),
                              dilation = np.array(self.conv2.dilation))
        self.fc1 = nn.Linear(out_shape[0]*out_shape[1]*self.conv2.out_channels, config.HIDDEN_SIZE)  # 6*6 from image dimension
        self.out = nn.Linear(config.HIDDEN_SIZE, config.OUT_SIZE)
    
    def get_shape(self,in_size,padding,kernal_size,stride,dilation=1):
        shape = (in_size+2*padding-dilation*(kernal_size-1)-1)/stride+1
        return shape.astype(np.int)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))   # In: (4, 105, 80)  Out: (16, 26, 20)
        x = F.relu(self.conv2(x))    # In: (16, 26, 20) Out: (32, 13, 10)
        x = x.view(x.size()[0], -1)  # In: (32, 13, 10) Out: (4160,)

        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class History(object):
    def __init__(self,config):
        self.stack_n = config.FRAME_STACK
        self.buffer = np.zeros((self.stack_n,config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]),dtype = config.IMG_DTYPE)
    
    def add(self,image):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1,:] = image
        
    def get(self):
        return torch.from_numpy(self.buffer.astype(np.float32))


def init_memory(env,buffer,initial_size,history_length):
    frame = env.new_game()
    for _ in np.arange(history_length):
        buffer.push(frame,0,0,False)
    for _ in tqdm(np.arange(initial_size)):
        frame,action,reward,done,info = env.random_step()
        buffer.push(frame,action,reward,done)
        if done:
            frame = env.new_game()
            for _ in np.arange(history_length):
                buffer.push(frame,0,0,False)


def preprocess(img,dtype = np.float32):
#    img_gray = np.mean(img, axis=2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_down = imresize(img_gray,(84,84),interpolation=cv2.INTER_AREA)
    img_norm = img_down/255.
    img_norm = np.asarray(img_norm,dtype = dtype)
    return img_norm          



def save_frames_as_gif(frames, path='./', filename='pong_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 50.0, frames[0].shape[0] / 50.0), dpi=144)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=0)
    anim.save(path + filename, writer='imagemagick', fps=12)


def train_step():
    if len(memory) < config.BATCH_SIZE:
        return None
    
    batch_state,batch_next_state,batch_action,batch_reward,batch_done = memory.sample()
    batch_state = batch_state.to(config.DEVICE)
    batch_next_state = batch_next_state.to(config.DEVICE)
    batch_action = batch_action.to(config.DEVICE)
    batch_reward = batch_reward.to(config.DEVICE)
    batch_done = batch_done.to(config.DEVICE)

    current_Q = Q(batch_state).gather(1, batch_action.unsqueeze(1).long())

    expected_Q = batch_reward.float()
    expected_Q[~batch_done] += config.GAMMA * target_Q(batch_next_state[~batch_done]).max(1)[0].detach()

    loss = F.mse_loss(current_Q, expected_Q.unsqueeze(1))
    #loss = F.smooth_l1_loss(current_Q, current_Q.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    for param in Q.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item(),current_Q.mean().item()

def predict(state,eps):
    if np.random.rand() < eps:
        return env.env.action_space.sample()
    q_vals = Q(state.to(config.DEVICE).unsqueeze(0)).squeeze()
    return q_vals.argmax().item()
    
if __name__ == "__main__":
    class ENVIROMENT_CONFIG(object):
        ENV_NAME = 'PongDeterministic-v4'
        RANDOM_START = 0
        IMG_DTYPE = np.float32
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class NN_CONFIG(ENVIROMENT_CONFIG):
        HIDDEN_SIZE = 256
        
    class DQN_CONFIG(NN_CONFIG):
        BASE = 1000
        BUFFER_SIZE = 200 * BASE 
        BATCH_SIZE = 32
        IMAGE_SIZE = (84,84)
        GAMMA = 1.0
        T_MAX = 5000
        EPISODE_MAX = 5000
        TARGET_UPDATE = 1*BASE
        EPS_0 = 1.0
        EPS_MIN = 0.1
        EPS_LEN = 2*BUFFER_SIZE
        INITIAL_COLLECTION=50 * BASE
        REPEAT_ACTIONS = 1
        FRAME_STACK = 4
        LEARNING_RATE = 1e-4
        SAVE_LATEST = 5
        
    config = DQN_CONFIG
    train_hist = []
    env = Enviroment(config)
    
    #device = torch.device('cpu')
    config.OUT_SIZE = env.env.action_space.n
    Q = DQN(config).to(config.DEVICE)
    ####### load model ##########
    #Q.load_state_dict(torch.load('pong_Q'))
    
    target_Q = DQN(config).to(config.DEVICE)
    target_Q.load_state_dict(Q.state_dict())
    target_Q.eval()
    memory = ReplayMemory(config)
    optimizer = optim.Adam(Q.parameters(),lr = config.LEARNING_RATE)    

    
    global_step = 0
    print("Begin initial replay memory collection.\n")
    init_memory(env,memory,config.INITIAL_COLLECTION,config.FRAME_STACK)
    frame_buffer = History(config)
    save_list = []
    print("Begin training.")
    for i_episode in range(config.EPISODE_MAX):
        tot_reward = 0
        frame = env.new_game()
        for _ in np.arange(config.FRAME_STACK):
            frame_buffer.add(frame)
            memory.push(frame,0,0,False)
        t_start = time.time()
        eps = max(config.EPS_MIN, config.EPS_0*(config.EPS_LEN-global_step)/config.EPS_LEN)
        for t in range(config.T_MAX):
            global_step+=1
            state = frame_buffer.get()
            action = predict(state,eps)
            cumulative_reward = 0
            for i in np.arange(config.REPEAT_ACTIONS):    
                frame, reward, done, info = env.step(action)
                cumulative_reward += reward
                if done:
                    break
            frame_buffer.add(frame)
            memory.push(frame, 
                        action,
                        cumulative_reward,
                        done)
            tot_reward += cumulative_reward
#            raise
            loss,q_val = train_step()
            if global_step % config.TARGET_UPDATE == 0:
                target_Q.load_state_dict(Q.state_dict())
                torch.save(Q.state_dict(), 'pong_Q%d'%(global_step))
                torch.save(target_Q.state_dict(), 'pong_Q_target_%d'%(global_step))
                save_list.append(('pong_Q%d'%(global_step),'pong_Q_target_%d'%(global_step)))
                if len(save_list) > config.SAVE_LATEST:
                    [os.remove(x) for x in save_list.pop(0)]
            if done:
                break  
                    
        train_hist += [tot_reward]
        print("Epoch:%d Global step:%d Loss:%.5f Q value: %.5f Total Reward:%.0f Trail Length:%d Epsilon:%.2F Elapsed Time:%.2f Buffer size:%d"%(i_episode, global_step, loss, q_val, tot_reward, t+1, eps, time.time() - t_start, len(memory)))
    ###
    
    plt.figure(figsize = (10,10))
    plt.plot(train_hist)
#    plt.plot(np.arange(0,EPISODE_MAX,10),
#             np.array(train_hist).reshape(-1, EPISODE_MAX).mean(axis = 1))
    plt.xlabel('# of Episode', fontsize = 20)
    plt.ylabel('Total Reward', fontsize = 20)


    def simulate(env, horizon,config, render = False):
        tot_reward = 0
        frame = env.new_game()
        frame_buffer = History(config)
        for _ in np.arange(config.FRAME_STACK):
            frame_buffer.add(frame)
        movie_frame = []
        for t in range(horizon):
            if render:
                #env.render()
                env.env.render()
                movie_frame.append(env.env.render(mode="rgb_array"))
                time.sleep(1/24)
                
            state = frame_buffer.get()
            action = predict(state,eps = 0)
            frame, reward, done, info = env.step(action)
            frame_buffer.add(frame)
            tot_reward += reward
            if done:
                break
                
        if render:    
            env.env.close()
                
        return tot_reward, reward, done, t, movie_frame
    
    ###
    env.random_start = 0
    _ = simulate(env, 100,config, True)
    torch.save(Q.state_dict(), 'pong_Q_final')
    torch.save(target_Q.state_dict(), 'pong_Q_target_final')

    Q = DQN(config).to(config.DEVICE)
    target_Q = DQN(config).to(config.DEVICE)
    ####### load model ##########
    
    Q.load_state_dict(torch.load('pong_Q_final'))
    target_Q.load_state_dict(torch.load('pong_Q_target_final'))
    
    reward_tot, reward, t, done, frames = simulate(env, 500, Q,config, True)
    save_frames_as_gif(frames[::2])
    


