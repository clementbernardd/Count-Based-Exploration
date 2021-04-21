import gym
import numpy as np
import torch
import torch.nn as nn
from tensordash.tensordash import Customdash
from RL_algorithm import *
from buffer import *
from dqn_agent import *
from qnetwork import *
from state import *
from simhash import *



class Train(RLAlgorithm):
  def __init__(self,env, dqagent, histories = None ) :
    '''
      - env : The openAI gym environment
      - dqagent : The agent used
      - histories : The histories from tensordash.tensordash

    '''
    super().__init__()
    self.env = env
    self.dqagent = dqagent

  def save(self) :
    self.dqagent.save()

  def load(self) :
    self.dqagent.load()

  def train(self, n_episodes = 100, eps = 0.1,t_max = 500 ,count_based = False, to_plot = True  ) :
    ''' Train with the agent algorithm '''
    self.plotting['Loss'] = []

    if count_based :
      eps = 0

    for e in range(n_episodes) :
      state = self.env.reset()

      all_epochs, all_rewards, all_loss  = 0, 0, 0

      for t in range(t_max) :
        # Select action with epsilon-greedy policy
        action = self.dqagent.act(state,eps)
        # Do the action in the environment
        new_state, reward, done, _ = self.env.step(action)
        loss = self.dqagent.step(state, action, reward, new_state, done,count_based)
        state = new_state

        if loss is not None :
          all_loss+=loss

        all_epochs+=1

        all_rewards+=reward

        if done :
          break
            # Add to the plotting
      self.plotting['Rewards'].append(all_rewards)
      self.plotting['Epochs'].append(all_epochs)
      self.plotting['Loss'].append(all_loss)

      losses = np.array(self.plotting['Loss'])
      rewards_ = np.array(self.plotting['Rewards'])


      if (e % 10 == 0) and to_plot :
        print('Episode {} , Reward : {}, Epsilon {}'.format(e,all_rewards, eps))
    # Save the model
    # self.dqagent.save()
