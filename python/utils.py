import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch

def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N, 'same')[(N-1):]


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


def plot_barchart_count_based(hash ,method ,env_name,  figsize = (14,6)):
  ''' Plot a barchart of the count of the states '''
  plt.subplots(figsize = figsize)
  plt.bar(np.arange(len(hash)), hash, 1.5, align = 'center' )
  plt.xlabel('States')
  plt.ylabel('Count of states')
  plt.title('Counts of states with {} on environment {}'.format(method, env_name))
  plt.grid(True)
  plt.show()



def train_all(algorithm, hp, n_episodes,env ,name_env,name , SEEDS , count_based = False, is_torch = False) :
  ''' Run the algorithm for different seeds '''
  all_means = {}
  all_count = {}
  for seed in SEEDS :
    if is_torch :
        torch.manual_seed(seed)
    np.random.seed(seed)
    hp['name'] = os.path.join(name_env, name, name+str(seed))
    if count_based :
      hp['name'] += '_cb'
    hp['env'] = env
    train = algorithm(**hp)
    train.train(n_episodes = n_episodes, count_based = count_based)
    all_means[seed] = train.plotting['Rewards']
    if seed == 77 :
      train.save()
    if count_based :
      all_count[seed] = train.hash

  path = os.path.join('results', name_env, name, name)
  path_hash = os.path.join('results', name_env, name, 'hash')
  if count_based :
    path+='_cb'
    save_obj(all_count, path_hash)
  save_obj(all_means,path)
  return all_means, all_count


def train_all_deep(algorithm, hp, n_episodes,env ,name_env,name , SEEDS , count_based = False) :
  ''' Run the algorithm for different seeds '''
  all_means = {}
  all_count = {}
  for seed in SEEDS :
    torch.manual_seed(seed)
    np.random.seed(seed)
    hp['name'] = os.path.join(name_env, name, name+str(seed))
    if count_based :
      hp['name'] += '_cb'
    hp['env'] = env
    train = algorithm(**hp)
    train.train(n_episodes = n_episodes, count_based = count_based)
    all_means[seed] = train.plotting['Rewards']
    if seed == 77 :
      train.save()
    if count_based :
      all_count[seed] = train.hash

  path = os.path.join('results', name_env, name, name)
  path_hash = os.path.join('results', name_env, name, 'hash')
  if count_based :
    path+='_cb'
    save_obj(all_count, path_hash)
  save_obj(all_means,path)
  return all_means, all_count


def get_hashes_mean(hash) :
  ''' Get the mean of the hashes representation '''
  size_max = np.max([ len(hash[key].values()) for key in hash  ])
  hashes = np.zeros((10, size_max))
  for i,x in enumerate(list(hash.values())) :
    current_length = len(x.values())
    hashes[i, :current_length]=np.array(list(x.values()))
  return hashes.mean(axis=0)


def train_all_deep(algorithm, hp, n_episodes,env ,name_env,name , SEEDS ,TRAIN, count_based = False) :
  ''' Run the algorithm for different seeds '''
  all_means = {}
  all_count = {}
  for i,seed in enumerate(SEEDS) :
    print('SEED {}/{}'.format(i,len(SEEDS)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    hp['name'] = os.path.join(name_env, name, name+str(seed))
    hp['state_size'] = len(env.observation_space.sample())
    hp['action_size'] = env.action_space.n
    if count_based :
      hp['name'] += '_cb'

    hp_train = {'env' : env, 'dqagent' : algorithm(**hp)}
    train = TRAIN(**hp_train)
    train.train(n_episodes = n_episodes, count_based = count_based, eps = 0.1, to_plot = False)
    all_means[seed] = train.plotting['Rewards']
    if seed == 77 :
      train.save()
    if count_based :
      all_count[seed] = train.dqagent.hash.hash

  path = os.path.join('results', name_env, name, name)
  path_hash = os.path.join('results', name_env, name, 'hash')
  if count_based :
    path+='_cb'
    save_obj(all_count, path_hash)
  save_obj(all_means,path)
  return all_means, all_count


def get_upper_bounds(results_by_seed) :
  ''' Return the means, upper bound and higher boud '''
  results = np.array(list(results_by_seed.values()))
  # Compute the average
  means = results.mean(axis = 0)
  # Compute the standard variation
  std = np.std(results, axis = 0)
  upper_b = means - std
  upper_h = means + std

  return means, upper_b, upper_h



def plot_rewards(all_means, all_means_cb, names , env_name, figsize = (14,8), nrows = 1, ncols = 1 ) :
  ''' Plot the rewards mean and std with and without count_based method '''
  fig, ax = plt.subplots(figsize = figsize, nrows = nrows, ncols = ncols)
  ax = np.array(ax)
  colors = ['b','orange']
  labels = ['Epsilon-greedy', 'Count-based']
  models = [all_means, all_means_cb]

  for i,name in enumerate(names) :
    for j in range(2) :
      means, upper_b, upper_h = get_upper_bounds(models[j][i])
      X = np.arange(len(means))
      ax.flatten()[i].plot(X, means, alpha = 0.6, color = colors[j])
      ax.flatten()[i].fill_between(X , upper_b, upper_h, color = colors[j], alpha = 0.3, label = labels[j])
      ax.flatten()[i].set_xlabel('Episode')
      ax.flatten()[i].set_ylabel('Rewards')
      ax.flatten()[i].grid(True)
      ax.flatten()[i].set_title(name)
      ax.flatten()[i].legend()
  fig.suptitle('Results with {} environment'.format(env_name), fontsize=15)


def create_files(ENVS, NAME_ALGORITHM) :
  for env in ENVS :
    models_path = os.path.join('models',env)
    results_path = os.path.join('results',env)
    if not os.path.exists(models_path) :
      os.mkdir(models_path)
    if not os.path.exists(results_path) :
      os.mkdir(results_path)
    models_path = os.path.join(models_path, NAME_ALGORITHM)
    results_path = os.path.join(results_path, NAME_ALGORITHM)
    if not os.path.exists(models_path) :
      os.mkdir(models_path)
    if not os.path.exists(results_path) :
      os.mkdir(results_path)




def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
