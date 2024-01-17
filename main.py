import pickle as pkl
import torch
import torch.nn as nn
import numpy as np
import sys
import re
import os
from datetime import datetime as time
from sklearn.decomposition import PCA

from general_util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "vivit" # Must be one of [base, enc, vivit]
caption_source = "deptree" # Must be one of [deptree, bert]
multiheaded = True
if multiheaded and caption_source != "deptree":
    print("Invalid setup - only deptree models can be multiheaded")
    exit()
if multiheaded:
  model_name = "%s_2head_plus"%(model_type)
else:
  model_name = "%s_%s"%(caption_source,model_type)

# Create a model of the correct type
if model_type == "vivit":
  from _vivit_classifier_util import *
  if multiheaded:
    MODEL = Classifier2HeadPlus()
  else:
    MODEL = ClassifierSmall_EgoOnly()
elif model_type == "enc":
  from _enc_class_util import *
  if multiheaded:
    MODEL = Classifier2HeadPlus()
  else:
    MODEL = Classifier()
elif model_type == "base":
  from _base_class_util import *
  if multiheaded:
    MODEL = Classifier2HeadPlus()
  else:
    MODEL = Classifier()
else:
  print("Invalid model type, must be one of base, enc, or vivit")
  exit()
MODEL.to(device)

# Load the SDLs from file
if caption_source == "bert":
  with open("./data/sdl_matrix.pkl","rb") as _in:
    sdl_embeddings = pkl.load(_in)
elif caption_source == "deptree":
  with open("./data/sdl_matrix_deptree.pkl","rb") as _in:
    sdl_embeddings = pkl.load(_in)
  with open("./data/sdl_deprels.pkl","rb") as _in:
    temp_deprels = pkl.load(_in)
  deprels = []
  for deprel in temp_deprels:
    while deprel[0] >= len(deprels):
      deprels.append(['']*15)
    deprels[deprel[0]][deprel[1]] = deprel[2]
else:
  print("Invalid caption source, must be either bert or deptree")
  exit()

# Load some manually defined SDLs
with open("./data/sdl_corrections.pkl","rb") as _in:
  action_corrections = pkl.load(_in)

# Randomize the data
np.random.seed(72)
np.random.shuffle(sdl_embeddings)

train_split = 0.9
split = int(train_split*len(sdl_embeddings))
n_epoch = 1
lr = 1e-5
ego_only = True

# Calculate the class weights so that the loss function is weighted by class
class_weights = torch.zeros(1,3).to(device)
for sdl in sdl_embeddings[:split]:
  action = torch.tensor(sdl[1][:,1],dtype=torch.int64)
  if caption_source == "deptree":
    second_action = torch.tensor(sdl[1][:,2],dtype=torch.int64)
  if sdl[0] in action_corrections:
    action,second_action = action_corrections[sdl[0]]
    action = torch.tensor([action_corrections[sdl[0]][0]],dtype=torch.int64)
    ac_gt = reduce_action(torch.tensor([action],dtype=torch.int64))
    if caption_source == "deptree":
      second_ac_gt = reduce_action(torch.tensor([second_action],dtype=torch.int64))
  elif ego_only:
    ac_gt = reduce_action(action[:1]).to(device) # Ego only
    if caption_source == "deptree":
      second_ac_gt = reduce_action(second_action[:1]).to(device)
  else:
    ac_gt = reduce_action(action).to(device)
    if caption_source == "deptree":
      second_ac_gt = reduce_action(second_action).to(device)
  for i in range(len(ac_gt[0])):
    if ac_gt[0][i] == 1: class_weights[0][i]+=1
    if caption_source == "deptree" and second_ac_gt[0][i] == 1: class_weights[0][i]+=1

# If using a vivit model, load the precomputed vivit outputs
if model_type == "vivit":
  video_urls = list(load_bddx_csv("./data/BDDX.csv")['InputVideo'])
  pattern = re.compile("/[^/]+/\w+-\w+\.mov")
  vivit_data_dir = "./post_vivit"

dist = BCELoss_ClassWeights
optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr)

# Train the model for n_epoch epochs
for epoch in range(n_epoch):
  train_loss = 0
  count = 0
  MODEL.train() # training
  for sdl in sdl_embeddings[:split]:
    optimizer.zero_grad()
    index = sdl[0]

    # Load the input
    if model_type == "vivit":
      video_loc = re.findall(pattern,video_urls[index[0]])[0][1:-4]
      post_vivit = "%s/%s-%i.data"%(vivit_data_dir,video_loc,index[1]+1)
      if not os.path.exists(post_vivit): continue
      nn_input = torch.load(post_vivit).to(device)
    else:
      c3d_vector = get_vectors_by_video(index[0],index[1])
      if type(c3d_vector) == type(None): continue
      c3d_vector = torch.tensor(c3d_vector).to(device)

    # Load the ground truth
    if index in action_corrections:
      action,second_action = action_corrections[index]
      action = torch.tensor([action_corrections[index][0]],dtype=torch.int64)
      if action == -1 and second_action == -1: continue
      ac_gt = reduce_action(torch.tensor([action],dtype=torch.int64)).to(device)
      if caption_source == "deptree":
        second_ac_gt = reduce_action(torch.tensor([second_action],dtype=torch.int64)).to(device)
    else:
      action = torch.tensor(sdl[1][:,1],dtype=torch.int64)
      if caption_source == "deptree":
        second_action = torch.tensor(sdl[1][:,2],dtype=torch.int64)
      if ego_only:
        ac_gt = reduce_action(action[:1]).to(device) # Ego only
        if caption_source == "deptree":
          second_ac_gt = reduce_action(second_action[:1]).to(device)
      else:
        ac_gt = reduce_action(action).to(device) # All actions
        if caption_source == "deptree":
          second_ac_gt = reduce_action(second_action).to(device)
    if all(ac_gt[0]==0): continue

    # Get the prediction from the model
    if model_type == "vivit":
      if multiheaded:
        ac_input = torch.tensor([[torch.argmax(ac_gt)]]).to(device)
        ac_pred = MODEL(nn_input,ac_input)
      else:
        ac_pred = MODEL(nn_input)
    else:
      if multiheaded:
        ac_pred = MODEL(c3d_vector,torch.argmax(ac_gt))
      else:
        ac_pred = MODEL(c3d_vector)

    # Compare the prediction to the ground truth
    if multiheaded:
      if all(second_ac_gt[0]==0): second_ac_gt = ac_gt # If no secondary action, set secondary to be same as primary
      loss = dist(ac_pred[0],ac_gt,class_weights)+dist(ac_pred[1],second_ac_gt,class_weights)
    elif caption_source == "bert":
      loss = 2*dist(ac_pred,ac_gt,class_weights)
    elif caption_source == "deptree":
      heuristic_wts = (4/3,2/3)
      loss = heuristic_wts[0]*dist(ac_pred,ac_gt,class_weights)+heuristic_wts[1]*dist(ac_pred,second_ac_gt,class_weights)

    train_loss+=loss.item()
    loss.backward()
    optimizer.step()
    count+=1

  print("Epoch %i training loss: %f"%(epoch+1,train_loss/count))
  torch.save(MODEL, "models/%s_%02i.model"%(model_name,epoch+1))

  MODEL.eval() # validation
  val_loss = 0
  val_count = 0
  for sdl in sdl_embeddings[split:]:
    index = sdl[0]

    # Load the input
    if model_type == "vivit":
      video_loc = re.findall(pattern,video_urls[index[0]])[0][1:-4]
      post_vivit = "%s/%s-%i.data"%(vivit_data_dir,video_loc,index[1]+1)
      if not os.path.exists(post_vivit): continue
      nn_input = torch.load(post_vivit).to(device)
    else:
      c3d_vector = get_vectors_by_video(index[0],index[1])
      if type(c3d_vector) == type(None): continue
      c3d_vector = torch.tensor(c3d_vector).to(device)

    # Load the ground truth
    if index in action_corrections:
      action,second_action = action_corrections[index]
      action = torch.tensor([action_corrections[index][0]],dtype=torch.int64)
      if action == -1 and second_action == -1: continue
      ac_gt = reduce_action(torch.tensor([action],dtype=torch.int64)).to(device)
      if caption_source == "deptree":
        second_ac_gt = reduce_action(torch.tensor([second_action],dtype=torch.int64)).to(device)
    else:
      #actor = torch.tensor(sdl[1][:,0],dtype=torch.int64)
      action = torch.tensor(sdl[1][:,1],dtype=torch.int64)
      if caption_source == "deptree":
        second_action = torch.tensor(sdl[1][:,2],dtype=torch.int64)
      #scene = torch.tensor(sdl[2],dtype=torch.int64)
      if ego_only:
        ac_gt = reduce_action(action[:1]).to(device) # Ego only
        if caption_source == "deptree":
          second_ac_gt = reduce_action(second_action[:1]).to(device)
      else:
        ac_gt = reduce_action(action).to(device) # All actions
        if caption_source == "deptree":
          second_ac_gt = reduce_action(second_action).to(device)
    if all(ac_gt[0]==0): continue

    # Get the prediction from the model
    if model_type == "vivit":
      if multiheaded:
        ac_input = torch.tensor([[torch.argmax(ac_gt)]]).to(device)
        ac_pred = MODEL(nn_input,ac_input)
      else:
        ac_pred = MODEL(nn_input)
    else:
      if multiheaded:
        ac_pred = MODEL(c3d_vector,torch.argmax(ac_gt))
      else:
        ac_pred = MODEL(c3d_vector)

    # Compare the prediction to the ground truth
    if multiheaded:
      if all(second_ac_gt[0]==0): second_ac_gt = ac_gt # If no secondary action, set secondary to be same as primary
      loss = dist(ac_pred[0],ac_gt,class_weights)+dist(ac_pred[1],second_ac_gt,class_weights)
    elif caption_source == "bert":
      loss = 2*dist(ac_pred,ac_gt,class_weights)
    elif caption_source == "deptree":
      heuristic_wts = (4/3,2/3)
      loss = heuristic_wts[0]*dist(ac_pred,ac_gt,class_weights)+heuristic_wts[1]*dist(ac_pred,second_ac_gt,class_weights)

    val_loss+=loss.item()
    val_count+=1

  print("Epoch %i validation loss: %f"%(epoch+1,val_loss/val_count))

