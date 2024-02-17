import tensorflow as tf
import moviepy.editor as moviepy
import torch
from math import isnan
import pickle as pkl
import struct
import os
import re
from general_util import load_bddx_csv

import sys
c3d_dir = "conv3d-video-action-recognition"
sys.path.insert(0,"%s/python"%(c3d_dir))

from data_prep import *
from mpypl_pipe_func import *
from mpypl_pipes import *
from c3dmodel import *

# t is the time of the first frame
# count is the number of frames
# clip_len is the number of seconds of the subclip
def sample_frames(clip,t=0,count=16,clip_len=1):
    output = []
    for i in range(count):
        time = t+clip_len*(i/count) # Sample at equal intervals across the subclip
        frame = clip.get_frame(time) # Grab the frame at the specified time
        output.append(frame)
    return torch.Tensor(output)

# clip_start and clip_end are the times of the BDD-X Sample
#   e.g., the first sample for video [0] has clip_start=0, clip_end=11
def get_frames(clip,clip_start,clip_end,frame_count=16,clip_len=1):
    output = []
    for i in range(clip_start,clip_end,clip_len):
        output.append( sample_frames(clip,t=i,count=frame_count,clip_len=clip_len).unsqueeze(0) )
    return output

MODEL = get_video_descriptor(weights_path='%s/models/weights_C3D_sports1M_tf.h5'%(c3d_dir))

video_dir = "/path/to/BDD/videos"
bddx = load_bddx_csv("./data/BDD-X-Annotations_v1.csv")
pattern = re.compile("/[^/]+/\w+-\w+\.mov")

print(bddx.head(1))

vector_size = 4096
fname = "vectors.txt"


with open(fname,"wb") as _out:
    pass # Create empty file

# Create c3d vectors file
vectors = np.zeros((60,vector_size)) # 60 is max vector size
for index,row in bddx.iterrows():
    url = row['InputVideo']
    url = re.findall(pattern,row['InputVideo'])[0][1:-4]
    clip = mp.VideoFileClip("%s/%s"%(video_dir,url),target_resolution=[112,112])
    for i in range(1,16):
        start = float(row['%iS'%(i)])
        finish = float(row['%iE'%(i)])
        if isnan(start) or isnan(finish):
            continue
        else:
            start = int(start)
            finish = int(finish)
        if finish == start: # Some clips have same start and finish times; avoid 0-second clips
            finish+=1
        elif start > finish:
            continue
        elif finish > 100: # Take care of some erroneous finish times
            while finish > 100:
                finish = int(finish/10)
        nn_inputs = get_frames(clip,start,finish)
        vsize = finish-start
        for j,nn_input in enumerate(nn_inputs):
            vectors[j] = MODEL.predict(tf.cast(nn_input,tf.float32))
        byte_vals = [bytearray(struct.pack('d',val)) for val in vectors[0:vsize].flatten()]
        with open(fname,"ab") as _out:
            for b in byte_vals:
                _out.write(b)
    clip.close()

# Create keys
keys_7x15 = []
keys_26x1 = []
vcount = 0

for index,row in bddx.iterrows():
    if index%500 == 0:
        print(index)
    keys_7x15.append([None]*15)
    for i in range(1,16):
        start = float(row['%iS'%(i)])
        finish = float(row['%iE'%(i)])
        if isnan(start) or isnan(finish):
            continue
        else:
            start = int(start)
            finish = int(finish)
        if finish == start: # Some clips have same start and finish times; avoid 0-second clips
            finish+=1
        elif start > finish: # Some clips have start after finish; ignore these
            continue
        elif finish > 100: # Take care of some erroneous finish times
            while finish > 100:
                finish = int(finish/10)
        vsize = finish-start
        keys_7x15[index-1][i-1] = (vcount,vsize)
        keys_26x1.append((vcount,vsize))
        vcount+=vsize

with open("vector_keys.pkl","wb") as _out:
    pkl.dump((keys_7x15,keys_26x1),_out)

