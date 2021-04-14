import torch
import torch.nn as nn


class STATE(nn.Module) :
  ''' State representation '''
  def __init__(self, input_size, output_size) :
    '''
      - input_size : The size of the state
      - output_size : The size of the new representation
    '''
    super(STATE,self).__init__()

    self.state = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.state(x.float())
