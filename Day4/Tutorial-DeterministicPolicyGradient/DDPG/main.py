import numpy as np 
import torch 
import gymnasium
import matplotlib.pyplot as plt 
from utils import *
from main_classes import * 
import os 
import argparse 


if __name__ == '__main__': 
    # Argument Parser.
    parser = argparse.ArgumentParser() 
    
    # Arg1 and arg2 relate to learning rates of critic and actor
    parser.add_argument('critic_lr', type = float) 
    parser.add_argument('actor_lr', type = float) 

    # Arg3 = destination folder for 
    parser.add_argument('save_dir')

    # Arg4 = type of noise for policy exploration.
    # 0 == Ornstein–Uhlenbeck / any other int = standard gaussian noise.
    parser.add_argument('noise',type = int)

    # Arg5 = EPISODES for training
    parser.add_argument('n_episodes')
    args = parser.parse_args() 

    # Required [box2d]
    env = gym.make('BipedalWalker-v3')
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is: ", dev)    
    EPISODES = int(args.n_episodes)

    hidden_layers = [400, 300]
    
    # Agent is created
    agent = DDPGagent(env, 
                      hidden_layers_dims=hidden_layers,                     #size of hidden layers for critic and actor NNs 
                      critic_lr=args.critic_lr, 
                      actor_lr=args.actor_lr, 
                      name_critic="critic_ddpg", 
                      name_actor="actor_ddpg", 
                      device=dev, 
                      directory=args.save_dir)
    
    name = "average_reward.png"

    # creates folder for NNs checkpoints, if not existing
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
        
    filename = os.path.join(args.save_dir,name)
    data_name = os.path.join(args.save_dir,"data.txt")
    best_score = env.reward_range[0]
    
    score_history = [] 
    normal_scalar = 0.24
    
    if(args.noise==0): 
        noise =  OUNoise(env.action_space) 
        
    step = 0
    for i in range(EPISODES): 
    
        state, _ = env.reset() 
        done = False 
        score = 0 
        
        while not done: 

            # deterministic action
            action = agent.get_action(state) 

            # plus UO noise or...
            if(args.noise==0): 
                action = noise.get_action(action, step) 
                step +=1

            # ... standard gaussian noise
            else : 
                action += np.random.randn(env.action_space.shape[0])*normal_scalar 
                normal_scalar *= 0.9987
                
            # step 
            new_state, reward, term, trunc, _ = env.step(action) 
            done = (term or trunc)
            # Cutoff in performance. Score -100 ---> episode ends!
            if reward <= -100: 
                reward = -1 
                
                # Replay buffer updated with step
                agent.update_replay_memory(state, action, reward, new_state, True)
                
            else: 
                # Replay buffer updated with step 
                agent.update_replay_memory(state, action, reward, new_state, done) 

            # Each step a single train. But not necessarily using last (s,a,r,s') tuple!)
            agent.train() 
            
            score += reward 
            state = new_state 
            
        score_history.append(score) 
        
        # Periodic I/O 
        if (i%50==0): 
            avg_score = np.mean(score_history[-100:])
            print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
            
            # Keeping track of best checkpoint so far.
            if avg_score > best_score: 
                best_score = avg_score 
                agent.save_model() 

    # Plotting scores 
    x = [i+1 for i in range(EPISODES)]
    plot_average_reward(x,score_history,filename)

    # Saving scores to text
    with open(data_name,'w') as f: 
        for i in range(0,len(score_history)): 
            f.write(str(score_history[i])+"\n")             
