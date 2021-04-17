import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def running_mean(x, N):

    mask=np.ones((1,N))/N
    mask=mask[0,:]
    result = np.convolve(x,mask,'same')

    return result


def plot_count(q_learning, q_learning_count, figsize = (14,8), window = 10, name = 'Q_learning') :

  fig, ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
  ax = np.array(ax)
  colors = ['g','orange']
  labels = ['Baseline', 'Count-based']
  models = [q_learning , q_learning_count]

  for i,label in enumerate(labels) :
    ax.flatten()[0].plot(models[i].plotting['Rewards'], alpha = 0.2, color =colors[i])
    ax.flatten()[0].plot(running_mean(models[i].plotting['Rewards'], window),color =colors[i], alpha=1, label = label)

    ax.flatten()[0].set_xlabel('Episode')
    ax.flatten()[0].set_ylabel('Rewards')
    ax.flatten()[0].grid(True)
    ax.flatten()[0].set_title(name)
    ax.flatten()[0].legend()

  return None


def plot_barchart_count_based(hash ,figsize = (14,6), method = 'Q-learning'):
  ''' Plot a barchart of the count of the states '''
  plt.subplots(figsize = figsize)
  plt.bar(np.arange(len(hash)), hash, 1.5, align = 'center' )
  plt.xlabel('States')
  plt.ylabel('Count of states')
  plt.title('Counts of states with {}'.format(method))
  plt.grid(True)
  plt.show()


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
