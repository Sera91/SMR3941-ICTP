import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import os 

MAX_ACTION = gym.make('BipedalWalkerHardcore-v3').action_space.high[0]

class ReplayBuffer(): 

    def __init__(self, max_size, input_shape, n_actions): 
        
        self.mem_size = max_size 
        self.mem_cntr = 0 
        
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) 
        self.action_memory = np.zeros((self.mem_size, n_actions)) 
        self.reward_memory = np.zeros(self.mem_size) 
        self.terminal_memory = np.zeros(self.mem_size, dtype = bool)
        
    def store_transition(self, state, action, reward, state_, done): 
        
        index = self.mem_cntr%self.mem_size 
        self.state_memory[index] = state
        self.new_state_memory[index] = state_ 
        self.terminal_memory[index] = done 
        self.reward_memory[index] = reward 
        self.action_memory[index] = action 
        self.mem_cntr +=1 
        
    def sample_buffer(self, batch_size): 
        
        max_mem = min(self.mem_cntr, self.mem_size) 
        batch = np.random.choice(max_mem, batch_size) 
        states = self.state_memory[batch] 
        states_ = self.new_state_memory[batch] 
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch] 
        dones = self.terminal_memory[batch] 
        return states, actions, rewards, states_, dones 
        
        
        
class CriticNetwork(nn.Module): 
        
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, name= None, chkpt_dir = "save_m_2"): 
        super(CriticNetwork, self).__init__ ()
        
        self.name = name
        if name is not None: 
            if not os.path.exists(chkpt_dir): 
                os.makedirs(chkpt_dir) 
            self.checkpoint_file= os.path.join(chkpt_dir,name +'_td3') 
        
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims 
        self.n_actions = n_actions 
        self.name = name 

        # The structure of the NN here is fixed, and has the actions as inputs both in the zeroth and first layer!
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims      + n_actions, self.fc2_dims)
        
        #scalar value of the critic (state-action value)
        self.q1 = nn.Linear(self.fc2_dims,1) 
        
    def forward(self, state, action): 
        """
        Calculates q(s,a)
        """
        q1_action_value = self.fc1(T.cat([state,action],dim=1))
        q1_action_value = F.relu(q1_action_value) 
        q1_action_value = self.fc2(T.cat([q1_action_value,action],dim=1))
        q1_action_value = F.relu(q1_action_value) 
        q1 = self.q1(q1_action_value) 
        return q1 

    # utility function to save model's checkpoint
    def save_checkpoint(self): 
        if self.name is not None:
            print("...saving...") 
            T.save(self.state_dict(),self.checkpoint_file)

    # utility function to load model's checkpoint
    def load_checkpoint(self): 
    
        if self.name is not None:
            print("..loading...") 
            self.load_state_dict(T.load(self.checkpoint_file)) 
        
class ActorNetwork(nn.Module): 
        
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, name=None, chkpt_dir = "save_m_2"): 
        
        super(ActorNetwork, self).__init__ ()
        
        self.name = name
        if name is not None: 
            if not os.path.exists(chkpt_dir): 
                os.makedirs(chkpt_dir) 
            self.checkpoint_file= os.path.join(chkpt_dir,name +'_td3') 

        self.input_dims = input_dims 
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims 
        self.n_actions = n_actions 
        self.name = name 
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)  
            
    def forward(self, state): 
        
        prob = self.fc1(state)
        prob = F.relu(prob) 
        prob = self.fc2(prob)
        prob = F.relu(prob) 
        mu = T.tanh(self.mu(prob))*MAX_ACTION
        return mu
        
    def save_checkpoint(self): 
        if self.name is not None:
            print("...saving...") 
            T.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self): 
        if self.name is not None:
            print("...loading...") 
            self.load_state_dict(T.load(self.checkpoint_file)) 
        
class Agent: 
    
    def __init__(self,
                 env, 
                 critic_lr = 0.001,
                 actor_lr = 0.001, 
                 tau = 0.005, 
                 gamma = 0.99, 
                 update_actor_interval = 2, 
                 warmup = 1000,
                 max_size = 1000000, 
                 layer1_size= 400, 
                 layer2_size= 300, 
                 batch_size = 100, 
                 noise = 0.2,
                 chkpt_dir = "model"):

        # infos from the environment
        self.input_dims = env.observation_space.shape
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0] 
        self.min_action = env.action_space.low[0]

        # gamma for discounted reward
        self.gamma = gamma 
        # tau for targets' parameters updates
        self.tau = tau 
    
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions) 
        self.batch_size = batch_size 
    
        self.learn_step_cntr = 0
        self.time_step = 0 
        self.warmup = warmup 

        # check local resources available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') 
    
        self.update_actor_iter = update_actor_interval

        # One actor NN
        self.actor = ActorNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "actor",chkpt_dir=chkpt_dir).to(self.device)

        # Two twin critics!
        self.critic_1 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "critic_1",chkpt_dir=chkpt_dir).to(self.device)
        self.critic_2 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "critic_2",chkpt_dir=chkpt_dir).to(self.device)

        # Targets NN, which will slowly follow to the learning ones.
        self.target_actor = ActorNetwork(self.input_dims, layer1_size, layer2_size, 
                                         self.n_actions, name = "target_actor",chkpt_dir=chkpt_dir).to(self.device)
        self.target_critic_1 = CriticNetwork(self.input_dims, layer1_size, layer2_size, 
                                             self.n_actions, name = "target_critic_1",chkpt_dir=chkpt_dir).to(self.device) 
        self.target_critic_2 = CriticNetwork(self.input_dims, layer1_size, layer2_size, 
                                             self.n_actions , name = "target_critic_2",chkpt_dir=chkpt_dir).to(self.device) 

        # Only the "true" actor and twin critics learn via gradient methods, the targets have a soft update of parameters
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = actor_lr) 
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr = critic_lr) 
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr = critic_lr) 
        
        self.noise = noise 
    
        self.update_network_parameters(tau=1)     
        
    def choose_action(self, observation, expl_noise, test=False): 
    
        if (self.time_step < self.warmup and (not test)): 
            mu = T.tensor(np.random.normal(scale = expl_noise, size = (self.n_actions,)))
        else: 
            self.actor.eval()
            state = T.tensor(observation, dtype = T.float).to(self.device)
            with T.no_grad():
                mu = self.actor.forward(state).to(self.device)             
            mu = mu +T.tensor(np.random.normal(0, self.max_action*expl_noise,size=self.n_actions), dtype = T.float).to(self.device) 
        
        # we have to climp to make sure that the actions are in the right boundaries, because adding the noise 
        # this could be not true 
        
        mu_prime = T.clamp(mu, self.min_action, self.max_action)
        self.time_step +=1 
        return mu_prime.cpu().detach().numpy()
        
    
    def store_transition(self, state, action, reward, new_state, done): 
        self.memory.store_transition(state, action, reward, new_state, done) 
        
    def train(self): 
        if self.memory.mem_cntr < self.batch_size: 
            return 
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size) 
        reward = T.tensor(reward, dtype= T.float).to(self.device) 
        done = T.tensor(done).to(self.device) 
        state = T.tensor(state, dtype= T.float).to(self.device) 
        action = T.tensor(action, dtype= T.float).to(self.device) 
        state_ = T.tensor(new_state, dtype= T.float).to(self.device)
        target_actions = self.target_actor.forward(state_) 
        
        noise = T.clamp(T.randn_like(action)*self.noise*self.max_action,0.5*self.min_action,0.5*self.max_action)
        target_actions = target_actions + noise
        target_actions = T.clamp(target_actions, self.min_action, self.max_action) 
        
        Q_tc1 = self.target_critic_1.forward(state_,target_actions) 
        Q_tc2 = self.target_critic_2.forward(state_,target_actions) 
        
        Q1 = self.critic_1.forward(state,action) 
        Q2 = self.critic_2.forward(state,action) 
        
        Q_tc1[done] = 0.0 
        Q_tc2[done] = 0.0 
        
        Q_tc1= Q_tc1.view(-1) 
        Q_tc2 = Q_tc2.view(-1) 
        
        critic_target_value = T.min(Q_tc1,Q_tc2) 
        
        target = reward +self.gamma*critic_target_value
        target = target.view(self.batch_size,1) 
        
        self.critic_1_optimizer.zero_grad() 
        self.critic_2_optimizer.zero_grad() 
        
        q1_loss = F.mse_loss(Q1,target) 
        q2_loss = F.mse_loss(Q2,target) 
        
        critic_loss = q1_loss + q2_loss 
        critic_loss.backward() 
        
        self.critic_1_optimizer.step() 
        self.critic_2_optimizer.step() 
        
        self.learn_step_cntr +=1 
        
        # update actor 
        
        if self.learn_step_cntr % self.update_actor_iter != 0: 
           return 
            
        self.actor_optimizer.zero_grad() 
        
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss) 
        actor_loss.backward() 
        
        self.actor_optimizer.step() 
        self.update_network_parameters() 
        
    def update_network_parameters(self, tau = None): 
    
        if tau is None: 
            tau = self.tau
        
        actor_params = self.actor.named_parameters() 
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        
        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)
        
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone()+ \
                                      (1-tau)*target_critic_1[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone()+ \
                                      (1-tau)*target_critic_2[name].clone()

        self.target_critic_2.load_state_dict(critic_2)
        
        for name in actor:
            actor[name] = tau*actor[name].clone()+ \
                                      (1-tau)*target_actor[name].clone()
                                      
        self.target_actor.load_state_dict(actor)
        
        
    def save_models(self): 
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        
        
    def load_models(self): 
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        
