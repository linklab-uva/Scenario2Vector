import tensorflow as tf
from math import isnan
import moviepy.editor as mp
import pandas as pd
import torch
import re
import os
from general_util import load_bddx_csv

import sys
sys.path.insert(0,'./scenic')
import scenic.projects.vivit.model as vivit
from scenic.projects.vivit.configs.kinetics400.vivit_base_k400 import get_config as get_vivit_config
from scenic.train_lib import optimizers
from scenic.train_lib.pretrain_utils import restore_pretrained_checkpoint
from scenic.train_lib.train_utils import restore_checkpoint, TrainState, initialize_model
import jax
import jax.numpy as jnp
import numpy as np

def load_ViViT(device='cpu'):
  # Load config file
  vivit_config = get_vivit_config()

  # Create model
  model_cls = vivit.get_model_cls(vivit_config.model_name)
  model = model_cls(vivit_config,dataset_meta_data={'num_classes':400})

  # Initialize random seed
  rng, init_rng = jax.random.split(jax.random.PRNGKey(42))

  print("Initializing Model") # TODO: Debug statement, remove before training
  # Initialize model
  input_shape = (None,32,224,224,3)
  (params, model_state, num_trainable_params,gflops) = initialize_model(
      model_def=model.flax_model,
      input_spec=[(input_shape,jnp.float32)],
      config=vivit_config,
      rngs=init_rng)
  optimizer = jax.jit(optimizers.get_optimizer(vivit_config).create, backend=device)(params)

  # Initialize train state
  train_state = TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=rng,
      accum_train_time=0)
  start_step = train_state.global_step

  print("Restoring from Checkpoint") # TODO: Debug statement, remove before training
  # Load train state from checkpoint
  restored_train_state = restore_pretrained_checkpoint('./checkpoints/checkpoint')

  # Restore model to checkpoint
  train_state = model.init_from_train_state(
      train_state=train_state,
      restored_train_state=restored_train_state,
      restored_model_cfg=vivit_config,
      restore_output_proj=True)

  # Define variable in model
  variables = {
  'params': train_state.optimizer.target,
  **train_state.model_state
  }

  return model,variables

# t is the time of the first frame
# count is the number of frames
# clip_len is the number of seconds of the subclip
def sample_frames(clip,t=0,count=32,clip_len=1):
  output = []
  for i in range(count):
    time = t+clip_len*(i/count) # Sample at equal intervals across the subclip
    frame = clip.get_frame(time) # Grab the frame at the specified time
    output.append(frame)
  return torch.Tensor(output)

# clip_start and clip_end are the times of the BDD-X Sample
#   e.g., the first sample for video [0] has clip_start=0, clip_end=11
def get_frames(clip,clip_start,clip_end,frame_count=32):
  output = []
  clip_len = clip_end-clip_start
  return sample_frames(clip,t=clip_start,count=frame_count,clip_len=clip_len).unsqueeze(0)

video_dir = "/path/to/BDD/videos"
data_out_dir = "./vivit_processed"

vivit,vivit_vars = load_ViViT()

if not os.path.exists(data_out_dir):
    os.makedirs("%s/samples-1k"%(data_out_dir))
    os.makedirs("%s/train"%(data_out_dir))
bddx = load_bddx_csv("./data/BDD-X-Annotations_v1.csv")
pattern = re.compile("/[^/]+/\w+-\w+\.mov")

for index,row in bddx.iterrows():
  url = row['InputVideo']
  dataset_name,video_name = re.findall(pattern,url)[0][1:-4].split("/")
  clip = mp.VideoFileClip("%s/%s/%s.mov"%(video_dir,dataset_name,video_name),target_resolution=[224,224])
  for i in range(1,16):
    out_loc = "%s/%s/%s-%i.data"%(data_out_dir,dataset_name,video_name,i)
    if os.path.exists(out_loc):
      continue
    start = float(row['%iS'%(i)])
    finish = float(row['%iE'%(i)])
    if isnan(start) or isnan(finish):
      continue
    else:
      start = int(start)
      finish = int(finish)
    if finish == start:
      finish+=1
    elif start > finish:
      continue
    elif finish > 100: # Take care of some erroneous finish times
      while finish > 100:
        finish = int(finish/10)
    nn_input = get_frames(clip,start,finish,frame_count=32)/255.
    preclassifier,prelogits = vivit.flax_model.apply(vivit_vars, vivit_input, train=False, mutable=False, debug=False)
    prelogits = prelogits.to_py()
    torch.save(torch.unsqueeze(torch.tensor(prelogits[0]),0),out_loc)
    out_file.close()
  clip.close()

