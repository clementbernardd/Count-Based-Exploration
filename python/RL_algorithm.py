import numpy as np
import matplotlib.pyplot as plt
from utils import * 

class Algorithm(object):
    pass
class RLAlgorithm(Algorithm):
    def __init__(self) :
      self.plotting = { 'Rewards' : [],'Epochs': []}
    def train(self):
        raise NotImplementedError
    def plot(self, figsize = (22,10), window = 10, name = 'Q-learning') :
      fig, ax = plt.subplots(figsize = figsize, nrows = 1, ncols = len(self.plotting))
      ax = np.array(ax)
      colors = ['b','g','r','c']
      metrics = list(self.plotting.keys())
      for i, metric in enumerate(metrics) :
        ax.flatten()[i].plot(self.plotting[metric], alpha = 0.3, color =colors[i])
        ax.flatten()[i].plot(running_mean(self.plotting[metric], window),color =colors[i], alpha=1)
        ax.flatten()[i].set_xlabel('Episode')
        ax.flatten()[i].set_ylabel(metric)
        ax.flatten()[i].grid(True)
        ax.flatten()[i].set_title('{} for {} algorithm'.format(metric,name))
