import numpy as np
import matplotlib.pyplot as plt
import gym
from RL_algorithm import *
from utils import *

class Q_learning(RLAlgorithm) :
  ''' Q-learning implementation '''
  def __init__(self, env, alpha,gamma, epsilon, n_actions = None , n_states = None ) :
    '''
      - env : The environment (an openai gym environment)
      - alpha : The step size between [0,1[
      - gamma : The discounted factor between [0,1[
      - epsilon : The epsilon-greedy factor between [0,1[
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


  def epsilon_greedy(self,state) :
    ''' Selection an action with the epsilon-greedy policy '''
    if np.random.uniform(0,1) < self.epsilon :
      # Return a random action
      return self.env.action_space.sample()
    else :
      # Return the best action from the state
      return np.argmax(self.q_table[state])
    return None

  def train(self, n_episodes = 1000) :
    ''' Train the model with Q-learning algorithm '''
    # Loop over the number of episodes
    for e in range(n_episodes) :
      # Reset the environment
      state = self.env.reset()

      epochs, rewards = 0,0

      done = False

      while not done :
        # Get an action with the epsilon-greedy policy
        action = self.epsilon_greedy(state)
        # Do the action in the environment
        new_state, reward, done, info = self.env.step(action)

        # Update the Q-table
        self.q_table[state, action] = self.q_table[state,action] + self.alpha * \
        (reward + self.gamma * np.max(self.q_table[new_state,:]) - self.q_table[state,action]   )
        # Update the state
        state = new_state

        epochs+=1
        rewards+=reward
      # Add to the plotting
      self.plotting['Rewards'].append(rewards)
      self.plotting['Epochs'].append(epochs)

    return None
