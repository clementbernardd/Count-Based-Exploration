import numpy as np
import matplotlib.pyplot as plt
import gym
from RL_algorithm import *
from utils import *


class SARSA(RLAlgorithm) :
  ''' SARSA implementation '''
  def __init__(self, env, alpha,gamma, epsilon, beta=None ,n_actions = None , n_states = None, name = 'Sarsa' ) :
    '''
      - env : The environment (an openai gym environment)
      - alpha : The step size between [0,1[
      - gamma : The discounted factor between [0,1[
      - epsilon : The epsilon-greedy factor between [0,1[
      - beta : The Beta factor from the count-based algorithm
      - n_actions : The size of the actions
      - n_states : The size of the states
    '''
    super().__init__()
    self.env = env
    self.n_actions = env.action_space.n  if n_actions is  None else n_actions
    self.n_states = env.observation_space.n if n_states is  None else n_states
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.q_table = np.zeros((self.n_states, self.n_actions))
    self.hash = np.zeros((self.n_states,1))
    self.beta = beta
    self.name = os.path.join('models', name)

  def save(self) :
    save_obj(self.q_table , self.name  + '_q_table')
    save_obj(self.hash , self.name  + '_hash' )

  def load(self)  :
    self.q_table = load_obj(self.name + '_q_table')
    self.hash = load_obj(self.name + '_hash')

  def epsilon_greedy(self,state,eps = 0) :
    ''' Selection an action with the epsilon-greedy policy '''
    if np.random.uniform(0,1) < eps :
      # Return a random action
      return self.env.action_space.sample()
    else :
      # Return the best action from the state
      return np.argmax(self.q_table[state])

  def train(self, n_episodes = 1000, count_based = False) :
    ''' Train the model with SARSA algorithm '''
    if count_based :
        eps = 0
    else :
        eps = self.epsilon
    # Loop over the number of episodes
    for e in range(n_episodes) :
      # Reset the environment
      state = self.env.reset()

      epochs, rewards = 0,0

      done = False

      action = self.epsilon_greedy(state)

      while not done :
        # Do the action in the environment
        new_state, reward, done, info = self.env.step(action)

        new_action = self.epsilon_greedy(new_state, eps)
        if count_based :
          # Update the count based method
          self.hash[state] = self.hash[state]+1
          # New reward
          count_reward =(self.beta / np.sqrt(self.hash[state]))[0]
        else :
          count_reward = 0
        new_reward = count_reward + reward
        # Update the Q-table
        self.q_table[state, action] = self.q_table[state,action] + self.alpha * \
        (new_reward + self.gamma * self.q_table[new_state, new_action] - self.q_table[state,action]   )
        # Update the state
        state = new_state
        action = new_action

        epochs+=1
        rewards+=reward
      # Add to the plotting
      self.plotting['Rewards'].append(rewards)
      self.plotting['Epochs'].append(epochs)

    return None
