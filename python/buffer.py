import torch
import numpy as np
import collections
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Buffer(object) :
  ''' Experience replay buffer '''
  def __init__(self,N ) :
    '''
      - N : Size of the buffer
    '''
    self.buffer = collections.deque(maxlen = N)
    self.N = N

  def add(self, states , actions, rewards, new_states, dones) :
    ''' Add experiences to buffer '''
    if isinstance(states, list) :
      for s,a,r,n,d in zip(states, actions, rewards, new_states, dones) :
        self.buffer.append([s,a,r,n,1*d])
    else :
      self.buffer.append([states , actions, rewards, new_states, 1*dones])

  def sample(self, batch_size) :
    ''' Return a sample of experiences in the buffer '''
    if batch_size >= len(self.buffer) :
      indexes = np.random.choice(np.arange(len(self.buffer)), batch_size, replace=True)
    else :
      indexes = np.random.choice(np.arange(len(self.buffer)), batch_size, replace=False)
    experiences = np.array(list(self.buffer))[indexes]

    states = torch.from_numpy(   np.stack(experiences[:,0], axis = 0)  ).to(device)
    actions = torch.from_numpy(np.stack(experiences[:,1], axis = 0)).to(device)
    rewards = torch.from_numpy(np.stack(experiences[:,2], axis = 0)).to(device)
    new_states = torch.from_numpy(np.stack(experiences[:,3], axis = 0)).to(device)
    dones = torch.from_numpy(np.stack(experiences[:,4], axis = 0)).to(device)

    return states, actions, rewards, new_states, dones
  def __len__(self) :
    return len(self.buffer)
