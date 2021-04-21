import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DuelingQNetwork(nn.Module) :
  ''' Dueling Network that uses Value and Advantage'''
  def __init__(self , state_size,hidden_size, action_space, name = 'dueling_dqn') :
    '''
      - state_emb : The size of the new state representation
      - hidden_size : The size of the hidden neural network
      - action_space : The output of the network (size of the actions)
      - name : The name of the model to save it

    '''
    super(DuelingQNetwork,self).__init__()

    self.checkpoint = os.path.join('models', name)

    self.state_size = state_size

    self.state = nn.Linear(state_size, hidden_size)

    self.value = nn.Linear(hidden_size, 1)

    self.advantage = nn.Linear(hidden_size, action_space)

  def forward(self, x) :
    x = (x.reshape(-1,self.state_size))

    x = nn.ReLU()(self.state(x.float()))

    value = self.value(x)

    advantage = self.advantage(x)

    q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))

    return q_value

  def save_checkpoint(self) :
      print('--- Save model checkpoint ---')
      torch.save(self.state_dict(), self.checkpoint)

  def load_checkpoint(self ) :
    print('--- Loading model checkpoint ---')
    if torch.cuda.is_available()  :
        self.load_state_dict(torch.load(self.checkpoint))
    else :
        self.load_state_dict(torch.load(self.checkpoint,map_location=torch.device('cpu')))
