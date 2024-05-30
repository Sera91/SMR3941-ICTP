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
    parser.add_argument('critic_lr', type = float) 
    parser.add_argument('actor_lr', type = float) 
    parser.add_argument('save_dir')
    parser.add_argument('n_episodes')
    args = parser.parse_args() 
    
    env = gym.make('BipedalWalkerHardcore-v3')
    agent = Agent(env, critic_lr = args.critic_lr, actor_lr = args.actor_lr, chkpt_dir = args.save_dir)
    
    EPISODES = int(args.n_episodes)
    
    name = "average_reward.png"
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
        
    filename = os.path.join(args.save_dir,name)
    data_name = os.path.join(args.save_dir,"data.txt")
    best_score = env.reward_range[0]
    
    score_history = [] 
    normal_scalar = 0.1
    expl_noise = 0.1
 
    for i in range(EPISODES): 
    
        state, _ = env.reset()
        done = False 
        score = 0 
        
        while not done: 
        
            action = agent.choose_action(state, expl_noise) 
            new_state, reward, term, trunc, _ = env.step(action) 
            done = (term or trunc)
            
            if reward <= -100: 
                reward =-1 
                agent.store_transition(state, action, reward, new_state, True) 
            else: 
                agent.store_transition(state, action, reward, new_state, False) 
                
            agent.train() 
            score += reward 
            state = new_state 
            
            
        score_history.append(score) 
        avg_score = np.mean(score_history[-50:]) 
        
        if(i%50==0): 

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
        
            print("Results during training procedure:") 
            print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
            
    x = [i+1 for i in range(EPISODES)]
    
    plot_average_reward(x, score_history,name)
    with open(data_name,'w') as f: 
        for i in range(0,len(score_history)): 
            f.write(str(score_history[i])+"\n")             
        
            
    
    
            
            
        
                
                
        
            
            
     
                
        
            
        
        
            
                
                
                
                
                
         
                
                
            
                
            
                
        
        
    
    
    
    
    
         
    
    
    
    
