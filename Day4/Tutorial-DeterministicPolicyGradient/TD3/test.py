import numpy as np 
import torch 
import gymnasium as gym
import matplotlib.pyplot as plt 
from utils import *
from main_classes2 import * 
import os 
import argparse 




if __name__ == '__main__': 

    parser = argparse.ArgumentParser() 
    parser.add_argument('save_dir')
    parser.add_argument('ep',type=int)
    parser.add_argument('render',type=int)
    
    args = parser.parse_args() 
    
    env = gym.make('BipedalWalkerHardcore-v3')
    env.seed(1)
    
    agent = Agent(env, chkpt_dir = args.save_dir) 
    agent.load_models() 
    
    EPISODES = args.ep
    reward_history = []
    
    for i in range(EPISODES): 
        state, _ = env.reset() 
        done = False 
        score = 0 
        if(args.render!=0):
            env.render()
            
        while not done: 
            action = agent.choose_action(state,0,test=True) 
            new_state, reward, term, trunc, _ = env.step(action) 
            done = (term or trunc)
            score += reward 
            
            if(args.render!=0):
                env.render()
            
            state = new_state
            
        print("Result during testing procedure = ") 
        print('episode ', i, 'score %.1f' % score)
        reward_history.append(score)
        
    rewards = np.asarray(reward_history) 
    average_reward = np.mean(rewards)
    max_reward = max(rewards) 
    min_reward = min(rewards) 
    print("Average reward = ", average_reward, ", max reward = ",max_reward,", min_reward = ",min_reward) 
    
        
        
        
        
            
        
            
