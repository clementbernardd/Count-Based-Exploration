import numpy as np
import matplotlib.pyplot as plt
import gym
from RL_algorithm import *
from utils import *


class ActionSpace(object) :
  def __init__(self, env) :
    self.env = env

  def sample(self) :
    return self.env.action_space.sample()


class Discretize(object) :
  ''' Class to discretize the state space '''
  def __init__(self, env, N ) :
    '''
      - env : The environment to discretize
      - N : the number of state to get (this won't be exactly the same )
    '''
    self.discrete_env = self.discretise_states(env, N)
    self.n_actions = env.action_space.n
    self.n_states = self.convert_state(env.observation_space.high)
    self.env = env
    self.action_space = ActionSpace(self.env)

  def step(self, action) :
    new_state, reward, done, info = self.env.step(action)
    return self.convert_state(new_state), reward, done, info

  def reset(self) :
    state = self.env.reset()
    return self.convert_state(state)

  def convert_state(self, state)  :
    ''' Return a number which tells the discrete state '''
    new_state = []
    for dim, s  in enumerate(state) :
      index_dim = np.where( s >= self.discrete_env[dim])[0][-1]
      new_state.append(index_dim)

    n_state = 0
    for dim in range(len(self.discrete_env)) :
      if dim != len(self.discrete_env) - 1 :
        n_state = new_state[dim] * len(self.discrete_env[dim])
      else :
        n_state+=new_state[dim]

    return n_state

  def discretise_states(self, env, N = 500) :
    ''' Discretize continuous state space for Q-learning '''
    high = env.observation_space.high
    low = env.observation_space.low
    state_size = len(env.observation_space.sample())
    k = int(np.power(N , 1/state_size))
    step = (high - low)/k

    discrete = {}

    for dim in range(state_size) :
      discrete[dim] = np.arange(low[dim], high[dim], step[dim])

    return discrete
