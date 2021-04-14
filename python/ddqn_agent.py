import torch
import torch.nn as nn
import numpy as np
from buffer import *
from qnetwork import *
from state import *
from simhash import *


class DDQNAgent(object) :
  ''' DDQN Agent '''
  def __init__(self,BUFFER_SIZE,state_size, state_emb , hidden_size,action_size, batch_size, \
               gamma, optimizer, criterion, lr ,device, UPDATE_EVERY, tau, k, beta, name = 'DDQN' ) :
    '''
    - BUFFER_SIZE : The size of the memory replay
    - state_size : The initial size of states
    - state_emb : The size of the embedding of the states
    - hidden_size : The size of the hidden layer in the QNetwork
    - action_size : The size of the action
    - batch_size : The size of the batch when sample from memory experience
    - gamma : The discount factor
    - optimizer : The optimizer to use
    - criterion : The criterion to use
    - lr : The learning rate
    - device : The device (cpu or gpu)
    - UPDATE_EVERY : The number of iteration before updating the local and target networks
    - tau : The tau factor to soft update the target and local network
    - k : The size of the A matrix in the Static hasing
    - beta : The beta parameter from the count-based exploration article
    - name : The name of the algorithm
    '''

    self.qnetwork_local = QNetwork(state_size, state_emb, hidden_size, action_size, name = name + '_local').to(device)
    self.qnetwork_target = QNetwork(state_size, state_emb, hidden_size, action_size, name = name + '_target').to(device)
    self.state = STATE(input_size = state_size, output_size = state_emb).to(device)
    self.hash = SimHash(state_emb, k) if k is not None else None

    self.optimizer = optimizer(self.qnetwork_local.parameters(), lr = lr )
    self.buffer = Buffer(BUFFER_SIZE)
    self.beta = beta

    self.batch_size = batch_size
    self.t_step = 0
    self.UPDATE_EVERY = UPDATE_EVERY
    self.gamma = gamma
    self.tau = tau
    self.action_size = action_size
    self.criterion = criterion
    self.device = device


  def step(self, state, action, reward, new_state, done, count_based = False ) :
    ''' Do one step of the algorithm '''
    self.buffer.add(state, action, reward, new_state, done)

    self.t_step = (self.t_step + 1)%self.UPDATE_EVERY

    if self.t_step == 0 :
      self.soft_update()

    if len(self.buffer) > self.batch_size :
      states, actions, rewards, new_states, dones = self.buffer.sample(self.batch_size)
      loss = self.learn(states, actions, rewards, new_states, dones, count_based)
      return loss

  def act(self, state, eps = 0) :
    ''' Choose action with epsilon-greedy policy '''
    self.qnetwork_local.eval()
    with torch.no_grad() :
      state = self.state(torch.from_numpy(state).to(self.device))
      all_actions = self.qnetwork_local(state)
    self.qnetwork_local.train()

    if np.random.uniform(0,1) < eps :
      return np.random.choice(np.arange(self.action_size))
    else :
      return torch.argmax(all_actions, dim =1 ).item()

  def learn(self, states, actions, rewards, new_states, dones, count_based = False) :
    ''' Update the QNetwork with mini-batch '''
    self.qnetwork_local.train()
    self.qnetwork_target.eval()

    states = self.state(states)
    new_states = self.state(new_states)


    preds = self.qnetwork_local(states).gather(1, actions.reshape(-1,1))

    with torch.no_grad() :
      next_preds = self.qnetwork_target(new_states).detach().max(1)[0]
      if count_based :
        # Add the counting based parts
        counts = self.hash.count(states)
        # Compute the new rewards
        count_reward = self.beta / torch.sqrt(counts)
      else :
        count_reward = 0
    # Get the reward with or without count-based method
    new_reward = count_reward + rewards
    target = new_reward + ((1 - dones) * self.gamma * next_preds)

    loss = self.criterion(preds.float(), target.reshape(-1,1).float()).to(self.device)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def soft_update(self) :
    ''' Do a soft update within the target and local networks '''
    for target_param, local_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()) :
      target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)
    # self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

  def save(self) :
    self.qnetwork_local.save_checkpoint()
    self.qnetwork_target.save_checkpoint()
  def load(self) :
    self.qnetwork_local.load_checkpoint()
    self.qnetwork_target.load_checkpoint()
