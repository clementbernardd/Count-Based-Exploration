import numpy as np
import torch


class SimHash(object) :
  def __init__(self, state_emb, k , device) :
    ''' Hashing between continuous state space and discrete state space '''
    self.hash = {}
    self.A = np.random.normal(0,1, (k , state_emb))
    self.device = device

  def count(self, states) :
    ''' Increase the count for the states and retourn the counts '''
    counts = []
    for state in states :
      key = str(np.sign(self.A @ state.detach().cpu().numpy()).tolist())
      if key in self.hash :
        self.hash[key] = self.hash[key] + 1
      else :
        self.hash[key] = 1
      counts.append(self.hash[key])

    return torch.from_numpy(np.array(counts)).to(self.device)
