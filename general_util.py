import numpy as np
import pickle as pkl
import pandas as pd
import torch
import struct

vector_size = 4096
double_size = 8

with open("data/vector_keys.pkl","rb") as _in:
    keys2d,keys1d = pkl.load(_in)

def get_vector_by_info(vector_info):
    """
    This function should only be called by get_vectors_by_video and get_vectors_by_clip
    """
    with open("data/vectors","rb") as _in:
        vector_loc = vector_info[0]*vector_size*double_size
        _in.seek(vector_loc,0)
        vectors_bytes = _in.read(double_size*vector_info[1]*vector_size)
        vectors_flat = struct.unpack('d'*vector_info[1]*vector_size,vectors_bytes)
        vectors = np.reshape(vectors_flat,(vector_info[1],vector_size))
    return vectors
    
def get_vectors_by_video(video_id,clip_id):
    """
    Inputs: video ID and clip ID. E.g., to get the first clip from the second video, which
    has the caption "The car is stopped. The car is at an intersection with a red light.",
    use get_vectors_by_video(1,0)
    """
    vector_info = keys2d[video_id][clip_id]
    if vector_info == None: return
    return get_vector_by_info(vector_info)

def get_vectors_by_clip(clip_id):
    """
    Input: clip ID in a flat format. E.g., to get the first clip from the second video, which
    has the caption "The car is stopped. The car is at an intersection with a red light.",
    """
    vector_info = keys1d[clip_id]
    if vector_info == None: return
    return get_vector_by_info(vector_info)

def reduce_action(gt):
  output_map = {"turn":0,"stop":1,"forward":2}
  input_map = ["turn","turn","none","turn","forward","forward","stop","stop","forward","none","stop","forward","none","forward","forward","forward","turn","turn","turn","none","none"]
  output = torch.zeros((1,len(output_map)))
  for action in gt:
    mapped_action = input_map[action]
    if mapped_action != "none":
      output[0][output_map[mapped_action]] = 1
  return output

# Code for this function taken from
# https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114/25
def BCELoss_ClassWeights(input, target, class_weights):
  input = torch.clamp(input,min=1e-7,max=1-1e-7)
  bce = -target*torch.log(input)-(1-target)*torch.log(1-input)
  weighted_bce = (bce*class_weights).sum(axis=1)/class_weights.sum(axis=1)[0]
  final_reduced_over_batch = weighted_bce.mean(axis=0)
  return final_reduced_over_batch

def load_bddx_csv(path):
  column_names = ['Index', 'InputVideo', '1S', '1E', '1A', '1J', '2S', '2E', '2A', '2J', '3S', '3E', '3A', '3J',
                  '4S', '4E', '4A', '4J','5S', '5E', '5A', '5J','6S', '6E', '6A', '6J','7S', '7E', '7A', '7J',
                  '8S', '8E', '8A', '8J','9S', '9E', '9A', '9J','10S', '10E', '10A', '10J','11S', '11E', '11A', '11J',
                  '12S', '12E', '12A', '12J','13S', '13E', '13A', '13J','14S', '14E', '14A', '14J','15S', '15E', '15A', '15J']

  bddx = pd.read_csv(path, names=column_names)
  bddx = bddx.drop(bddx.index[0])
  return bddx
