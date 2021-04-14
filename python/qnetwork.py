import os
import torch
import torch.nn as nn


class QNetwork(nn.Module) :
  ''' Representation of the Q matrix '''
  def __init__(self, state_size , state_emb,hidden_size, action_space, name = 'dqn') :
    '''
      - state_size : The size of the origin state representation
      - state_emb : The size of the new state representation
      - hidden_size : The size of the hidden neural network
      - action_space : The output of the network (size of the actions)
      - name : The name of the model to save it 
    '''
    super(QNetwork,self).__init__()

    self.checkpoint = os.path.join('models', name)

    self.state_emb = state_emb
    self.fc = nn.Sequential(
          nn.Linear(state_emb,hidden_size ),
          nn.ReLU(),
          nn.Linear(hidden_size, action_space),
    )

  def forward(self, x) :
    x = x.reshape(-1,self.state_emb)
    return self.fc(x)

  def get_state(self,x) :
    return self.state(x)

  def save_checkpoint(self) :
      print('--- Save model checkpoint ---')
      torch.save(self.state_dict(), self.checkpoint)

  def load_checkpoint(self ) :
    print('--- Loading model checkpoint ---')
    if torch.cuda.is_available()  :
        self.load_state_dict(torch.load(self.checkpoint))
    else :
        self.load_state_dict(torch.load(self.checkpoint,map_location=torch.device('cpu')))
