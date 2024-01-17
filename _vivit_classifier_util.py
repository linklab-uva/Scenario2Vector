import torch
import torch.nn as nn
import torch.nn.functional as F

from general_util import *

class ClassifierBig(nn.Module):
  def __init__(self):
    super(ClassifierBig, self).__init__()
    self.input_size = 768
    self.hidden_size = 64
    self.num_classes = 3

    self.linear1 = nn.Linear(self.input_size,self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size,self.num_classes)

    self.linear1.apply(self.init_weights)
    self.linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input):
    x = _input[:, 0]
    x = torch.relu(self.linear1(x))
    x = torch.sigmoid(self.linear2(x),dim=1)
    return x

  def predict(self, _input):
    _out = self.forward(_input)
    return (_out>0.5).float()

class ClassifierSmall(nn.Module):
  def __init__(self):
    super(ClassifierSmall, self).__init__()
    self.input_size = 768
    self.hidden_size = 64
    self.num_classes = 3

    self.linear1 = nn.Linear(self.input_size,self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size,self.num_classes)

  def forward(self, _input):
    x = F.relu(self.linear1(_input))
    y = torch.sigmoid(self.linear2(x))
    x = self.linear2(x)
    x = torch.sigmoid(x)
    return x

  def predict(self, _input):
    _out = self.forward(_input)
    return (_out>0.5).float()

class ClassifierSmall_EgoOnly(nn.Module):
  def __init__(self):
    super(ClassifierSmall_EgoOnly, self).__init__()
    self.input_size = 768
    self.hidden_size = 64
    self.num_classes = 3

    self.linear1 = nn.Linear(self.input_size,self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size,self.num_classes)

    self.linear1.apply(self.init_weights)
    self.linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input):
    x = F.relu(self.linear1(_input))
    x = F.softmax(self.linear2(x),dim=1)
    return x

  def predict(self, _input):
    _out = self.forward(_input)
    return torch.argmax(_out)

class Classifier2Head(nn.Module):
  def __init__(self):
    super(Classifier2Head, self).__init__()
    self.input_size = 768
    self.hidden_size = 64
    self.num_classes = 3

    self.head1_linear1 = nn.Linear(self.input_size,self.hidden_size)
    self.head1_linear2 = nn.Linear(self.hidden_size,self.num_classes)

    self.head2_linear1 = nn.Linear(self.input_size,self.hidden_size)
    self.head2_linear2 = nn.Linear(self.hidden_size,self.num_classes)

    self.head1_linear1.apply(self.init_weights)
    self.head1_linear2.apply(self.init_weights)
    self.head2_linear1.apply(self.init_weights)
    self.head2_linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input):
    y1 = F.relu(self.head1_linear1(_input))
    y1 = torch.sigmoid(self.head1_linear2(y1))

    y2 = F.relu(self.head2_linear1(_input))
    y2 = torch.sigmoid(self.head2_linear2(y2))
    return (y1,y2)

  def predict(self, _input):
    p1,p2 = self.forward(_input)
    p1 = torch.argmax(p1)
    p2 = torch.argmax(p2)
    return (p1,p2)

class Classifier2HeadPlus(nn.Module):
  def __init__(self):
    super(Classifier2HeadPlus, self).__init__()
    self.input_size = 768
    self.hidden_size = 64
    self.num_classes = 3

    self.head1_linear1 = nn.Linear(self.input_size,self.hidden_size)
    self.head1_linear2 = nn.Linear(self.hidden_size,self.num_classes)

    self.head2_linear1 = nn.Linear(self.input_size+1,self.hidden_size)
    self.head2_linear2 = nn.Linear(self.hidden_size,self.num_classes)

    self.head1_linear1.apply(self.init_weights)
    self.head1_linear2.apply(self.init_weights)
    self.head2_linear1.apply(self.init_weights)
    self.head2_linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input, gt_input):
    y1 = F.relu(self.head1_linear1(_input))
    y1 = torch.sigmoid(self.head1_linear2(y1))

    y2 = F.relu(self.head2_linear1(torch.cat([_input,gt_input],dim=1)))
    y2 = torch.sigmoid(self.head2_linear2(y2))
    return (y1,y2)

  def predict(self, _input):
    p1 = F.relu(self.head1_linear1(_input))
    p1 = torch.sigmoid(self.head1_linear2(p1))
    p1 = torch.argmax(p1)

    gt_input = torch.tensor([[p1]])
    p2 = F.relu(self.head2_linear1(torch.cat([_input,gt_input],dim=1)))
    p2 = torch.sigmoid(self.head2_linear2(p2))
    p2 = torch.argmax(p2)
    return (p1,p2)

