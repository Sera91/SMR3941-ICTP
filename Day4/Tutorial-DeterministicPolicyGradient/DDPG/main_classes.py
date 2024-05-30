import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.autograd 
import torch.optim as optim 

import numpy as np 
import gymnasium as gym
from collections import deque 
import random
import os

# Ornstein–Uhlenbeck noise.
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    # the internal state evolution
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        # final action is the result between the determinist part "action" and the (correlated) noise "ou_state", which has a variance which decays in time.
        return np.clip(action + ou_state, self.low, self.high)
        
        
        
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
        
        
# class to store memories as tuples (s,a,r,s',done) 
# with functions to add memories, remove memories and sample from those.
class Replay_Memory: 
    
    def __init__(self, max_size): 
        
        self.max_size = max_size 
        # deque is a fast list-like type
        self.buffer = deque(maxlen = max_size)

    
    def push(self, state, action, reward, next_state, done): 
        """
        push: receives a new experience (s,a,r,s',done) to add to the buffer 
        """
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """
        sample: returns a (random) batch of size batch_size of experiences from the buffer 
        """
        state_batch = []
        action_batch = [] 
        reward_batch = [] 
        next_state_batch = [] 
        done_batch = [] 
        batch = random.sample(self.buffer, batch_size)
        
        for experience in batch: 
            
            state, action, reward, next_state, done = experience 
    
            state_batch.append(state)        
            action_batch.append(action) 
            reward_batch.append(reward) 
            next_state_batch.append(next_state)
            done_batch.append(not done)
            
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch 
    
    def __len__(self): 
        
        return len(self.buffer)
        
        


class Critic(nn.Module): 
    
    def __init__(self, input_dim, hidden_layers_dims, n_actions, name=None, chkpt="model"): 
        
        super(Critic, self).__init__() 
        
        # hidden_layers_dims is a list with dimension of all hidden layers.
        # number of hidden_layers is inferred by its length
        self.hidden_layers_dims = list(input_dim) + hidden_layers_dims
        self.n_actions = n_actions 
        
        # name of model to save
        if name is not None:
            if not os.path.exists(chkpt): 
                os.makedirs(chkpt)
            self.filename = os.path.join(chkpt, name +'_ddpg')

        # hidden_layers are linear + layernorm. 
        self.hidden_layers_list = []
        for dim_in, dim_out in zip(self.hidden_layers_dims[:-1],self.hidden_layers_dims[1:]): 
            self.hidden_layers_list.append( nn.Linear( dim_in, dim_out) )
            self.hidden_layers_list.append( nn.LayerNorm(dim_out) )
            self.hidden_layers_list.append( nn.ReLU() )

        last_hidden_layer_dim = self.hidden_layers_dims[-1]
        self.hidden_layers_state_value = nn.Sequential(*self.hidden_layers_list)
        
        # a linear layer is constructed from the actions directly to the last hidden layer 
        self.action_value = nn.Linear(self.n_actions, last_hidden_layer_dim)
                                      
        # this is the final layer which returns the quality function q(s,a)
        self.q = nn.Linear(last_hidden_layer_dim,1)
        
    def forward(self, state, action): 
        
        # first "branch" from input state to final hidden layer
        state_value = self.hidden_layers_state_value(state)
        
        # second "branch" from action to final hidden layer
        action_value = F.relu(self.action_value(action))

        # merge of the two branches
        state_action_value = F.relu(torch.add(state_value, action_value))
        
        # evaluation of q(s,a)
        state_action_value = self.q(state_action_value)
        return state_action_value 
    
    # q is initialized to smaller values than what "suggested" by the 1/sqrt(input_dim) rule
    def init_weights(self): 
        
        init_weights_q = 0.003
        torch.nn.init.uniform_(self.q.weight.data, -init_weights_q, init_weights_q)
        torch.nn.init.uniform_(self.q.bias.data,   -init_weights_q, init_weights_q)
        
    # saves checkpoint for the model
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.filename)
        print("saving")

    # loads checkpoint for the model
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.filename))
    



class Actor(nn.Module):
    
    def __init__(self, input_dim, hidden_layers_dims, n_actions, name=None, chkpt="model"): 
        
        super(Actor, self).__init__() 
            
        # hidden_layers_dims is a list with dimension of all hidden layers.
        # number of hidden_layers is inferred by its length
        self.hidden_layers_dims = list(input_dim) + hidden_layers_dims
        self.n_actions = n_actions 
        
        # name of model to save
        if name is not None:
            if not os.path.exists(chkpt): 
                os.makedirs(chkpt)
            self.filename = os.path.join(chkpt, name +'_ddpg')

        # hidden_layers are linear + layernorm. 
        self.hidden_layers_list = []
        for dim_in, dim_out in zip(self.hidden_layers_dims[:-1],self.hidden_layers_dims[1:]): 
            self.hidden_layers_list.append( nn.Linear( dim_in, dim_out) )
            self.hidden_layers_list.append( nn.LayerNorm(dim_out) )
            self.hidden_layers_list.append( nn.ReLU() )

        last_hidden_layer_dim = self.hidden_layers_dims[-1]
        # a linear layer is constructed from the last hidden layer to the action  
        self.hidden_layers_list.append( nn.Linear(last_hidden_layer_dim, self.n_actions) )
        
        # creates a torch Module, to make parameters visible to torch optimizer
        self.hidden_layers = nn.Sequential(*self.hidden_layers_list)

    def forward(self,state):
        
        # from input state to final hidden layer to squashed mu
        x = torch.tanh( self.hidden_layers( state) )
        return x
    
    # here for symmetry reasons
    def init_weights(self): 
        pass
        
        
    # saves checkpoint for the model
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.filename)
        print("saving")

    # loads checkpoint for the model
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.filename))

    # loads checkpoint for the model
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.filename))
        
        
        
class DDPGagent: 
    """
    Initializes an agent which uses the Deep Deterministic Policy Gradient agent 
    """
    def __init__(self, 
                 env, 
                 hidden_layers_dims,
                 replay_min=100,
                 replay_size=1000000,
                 critic_lr=0.00015, 
                 actor_lr=0.000015, 
                 tau =0.001, 
                 gamma=0.99,
                 loss=nn.MSELoss(), 
                 batch_size=64, 
                 name_critic=None, 
                 name_actor=None, 
                 device = "cpu",
                 directory = "models"):
        
        # "reads" the environment and sets
        self.env = env 
        self.input_dim = env.observation_space.shape 
        self.n_actions = env.action_space.shape[0]
        self.tau = tau 
        self.device = device
        self.gamma = gamma 
        
        # 
        self.batch_size = batch_size 
        self.memory= Replay_Memory(replay_size)
        self.replay_min = replay_min
        
        # sets the names used for folder creation and checkpoints saves
        self.name_critic = name_critic
        self.name_actor = name_actor 
        
        # creates critic
        self.critic = Critic(self.input_dim, hidden_layers_dims, self.n_actions, name=name_critic, chkpt=directory).to(device)
        name_target_critic = None
        
        if name_critic is not None: 
            name_target_critic = name_critic + "_target"
        
        # creates target critic
        self.target_critic = Critic(self.input_dim, hidden_layers_dims, self.n_actions, name = name_target_critic,chkpt=directory).to(device)
        
        
        # creates actor
        self.actor = Actor(self.input_dim, hidden_layers_dims, self.n_actions, name = name_actor,chkpt=directory).to(device)
        
        name_target_actor = None 
        if name_actor is not None: 
                name_target_actor = name_actor + "_target"
            
        # creates target actor
        self.target_actor = Actor(self.input_dim, hidden_layers_dims, self.n_actions, name = name_target_actor,chkpt=directory).to(device)
        
        # initialization of weights (possibly redundant)
        self.critic.init_weights()
        self.actor.init_weights()

        # initialization of weights (possibly redundant)
        self.update_target_weights()
        
        self.critic_criterion = loss 
        self.actor_criterion = loss  
        
        # initialization of optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr,weight_decay=0.01)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
    
    # explict decay of learning rates
    def update_critic_optimizer(self, learning_rate):
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate)
        
    def update_actor_optimizer(self, learning_rate): 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        
    
    def update_replay_memory(self,state, action, reward, next_state, done): 
        """ Adds single experience to the memory buffer.
        Receives s,a,r,s',done
        """
        self.memory.push(state, action, reward, next_state, done)    

        
    def update_target_weights(self,tau=1): 
        """ Soft-update of the target networks towards the current (learned) network.
        """
        
        for target_param, param in zip(self.target_critic.parameters(),self.critic.parameters()): 
            target_param.data.copy_(param.data * self.tau + target_param.data *(1.0 - tau))
            
        for target_param, param in zip(self.target_actor.parameters(),self.actor.parameters()): 
            target_param.data.copy_(param.data * self.tau + target_param.data *(1.0 - tau))
            
            
    def get_action(self, observation): 
        """ From state (observation) to the deterministic (+noise) action
        """
        self.actor.eval()  #because I have batch norm 
        
        observation = torch.tensor(observation, dtype= torch.float).to(self.device)
        actor_action = self.actor(observation)
        action = actor_action.cpu().detach().numpy()  
        
        return action 
    
    
    def train(self): 
        
        # training starts only after some sampling has been done
        if len(self.memory) <  self.replay_min:
            return 
        
        # randomly sampled experience from past.
        states, actions, rewards, next_states, not_done = self.memory.sample(self.batch_size)
        
        states = torch.tensor(np.array(states), dtype = torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype = torch.float).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype = torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype = torch.float).to(self.device)
        not_done = torch.tensor(np.array(not_done)).unsqueeze(1).to(self.device)
        
        # 
        self.actor.eval()     
        self.critic.eval() 
        

        self.target_actor.eval() 
        self.target_critic.eval() 
        
        # a' = mu(s')
        # target q -> q(s', mu)
        target_actions = self.target_actor.forward(next_states)
        target_critic_value = self.target_critic(next_states, target_actions) 
    
        # Q^exp = r + gamma Q(s',a')
        targets = rewards + self.gamma*not_done*target_critic_value
        targets.to(self.device)          
        
        self.critic.train()
        self.critic_optimizer.zero_grad()
        critic_value = self.critic.forward(states, actions)
        
        # if MSE, loss = (r + gamma Q(s', a') - Q(s,a))^2
        loss = self.critic_criterion(critic_value, targets)
        loss.backward() 
        
        self.critic_optimizer.step() 
        self.critic.eval() 
        
        self.actor_optimizer.zero_grad() 
        self.actor.train() 
        
        mu = self.actor.forward(states)
        

        # gradient is performed directly on the Q(s,mu)
        # the minus sign transforms the minimization (implied in the optimizer) with the 
        # maximization objective of the deterministic policy gradient
        actor_loss = -self.critic.forward(states,mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.update_target_weights(self.tau)
        
    def save_model(self):
        """ Saves all models' checkpoints in folder given by names
        """
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_model(self):
        """ Loads all models' checkpoints in folder given by names
        """
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
