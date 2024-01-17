import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import sys
from torchvision.transforms import Normalize

from general_util import *

class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.num_actors = 3 # Actually output size, but don't want to rename for compatability reasons
    self.hidden_size = 512
    self.input_size = 512
    self.max_size = 60

    with open("data/pca_%i_model.pkl"%(self.input_size),"rb") as _in:
      self.pca = pkl.load(_in)
    sample = get_vectors_by_video(0,0)
    self.norm = Normalize([sample[0].mean()],[sample[0].std()])
    self.linear1 = nn.Linear(self.max_size*self.input_size,self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size,self.num_actors) # -1 if no ego

    self.linear1.apply(self.init_weights)
    self.linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input):
    x = torch.zeros(1,self.max_size,self.input_size)
    _in = self.pca.transform(_input) # Reduce dimensionality
    x[0][0:_in.shape[0]] = torch.tensor(_in)
    x = self.norm(x).detach()
    x = torch.flatten(x,1,2)

    x = F.relu(self.linear1(x))
    x = torch.sigmoid(self.linear2(x))
    return x

  def predict(self, _input):
    _out = self.forward(_input)
    return torch.argmax(_out)

class Classifier2Head(nn.Module):
  def __init__(self):
    super(Classifier2Head, self).__init__()
    self.num_actors = 3 # Actually output size, but don't want to rename for compatability reasons
    self.hidden_size = 512
    self.input_size = 512
    self.max_size = 60

    with open("data/pca_%i_model.pkl"%(self.input_size),"rb") as _in:
      self.pca = pkl.load(_in)
    sample = get_vectors_by_video(0,0)
    self.norm = Normalize([sample[0].mean()],[sample[0].std()])
    self.head1_linear1 = nn.Linear(self.max_size*self.input_size,self.hidden_size)
    self.head1_linear2 = nn.Linear(self.hidden_size,self.num_actors) # -1 if no ego
    self.head2_linear1 = nn.Linear(self.max_size*self.input_size,self.hidden_size)
    self.head2_linear2 = nn.Linear(self.hidden_size,self.num_actors) # -1 if no ego

    self.head1_linear1.apply(self.init_weights)
    self.head1_linear2.apply(self.init_weights)
    self.head2_linear1.apply(self.init_weights)
    self.head2_linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input):
    x = torch.zeros(1,self.max_size,self.input_size)
    _in = self.pca.transform(_input) # Reduce dimensionality
    x[0][0:_in.shape[0]] = torch.tensor(_in)
    x = self.norm(x).detach()
    x = torch.flatten(x,1,2)

    y1 = F.relu(self.head1_linear1(x))
    y1 = torch.sigmoid(self.head1_linear2(y1))

    y2 = F.relu(self.head2_linear1(x))
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
    self.num_actors = 3 # Actually output size, but don't want to rename for compatability reasons
    self.hidden_size = 512
    self.input_size = 512
    self.max_size = 60

    with open("data/pca_%i_model.pkl"%(self.input_size),"rb") as _in:
      self.pca = pkl.load(_in)
    sample = get_vectors_by_video(0,0)
    self.norm = Normalize([sample[0].mean()],[sample[0].std()])
    self.head1_linear1 = nn.Linear(self.max_size*self.input_size,self.hidden_size)
    self.head1_linear2 = nn.Linear(self.hidden_size,self.num_actors) # -1 if no ego
    self.head2_linear1 = nn.Linear(self.max_size*self.input_size+1,self.hidden_size)
    self.head2_linear2 = nn.Linear(self.hidden_size,self.num_actors) # -1 if no ego

    self.head1_linear1.apply(self.init_weights)
    self.head1_linear2.apply(self.init_weights)
    self.head2_linear1.apply(self.init_weights)
    self.head2_linear2.apply(self.init_weights)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      if m.bias != None: m.bias.data.fill_(0.01)

  def forward(self, _input, primary_gt):
    x = torch.zeros(1,self.max_size,self.input_size)
    _in = self.pca.transform(_input) # Reduce dimensionality
    x[0][0:_in.shape[0]] = torch.tensor(_in)
    x = self.norm(x).detach()
    x = torch.flatten(x,1,2)

    y1 = F.relu(self.head1_linear1(x))
    y1 = torch.sigmoid(self.head1_linear2(y1))

    gt_input = torch.tensor([[primary_gt]])
    y2 = F.relu(self.head2_linear1(torch.cat([x,gt_input],dim=1)))
    y2 = torch.sigmoid(self.head2_linear2(y2))

    return (y1,y2)

  def predict(self, _input):
    x = torch.zeros(1,self.max_size,self.input_size)
    _in = self.pca.transform(_input) # Reduce dimensionality
    x[0][0:_in.shape[0]] = torch.tensor(_in)
    x = self.norm(x).detach()
    x = torch.flatten(x,1,2)

    p1 = F.relu(self.head1_linear1(x))
    p1 = torch.sigmoid(self.head1_linear2(p1))
    p1 = torch.argmax(p1)

    gt_input = torch.tensor([[p1]])
    p2 = F.relu(self.head2_linear1(torch.cat([x,gt_input],dim=1)))
    p2 = torch.sigmoid(self.head2_linear2(p2))
    p2 = torch.argmax(p2)

    return (p1,p2)
