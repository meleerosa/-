#!/usr/bin/env python
# coding: utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task')
parser.add_argument('--gpu')
args = parser.parse_args()
# In[1]:


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



# In[3]:


import torch, detectron2


# In[4]:


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[5]:


task = args.task
root_dir = '/data2/intflow/'


# In[6]:


# # keypoint numbering vis

# json_path = '/data2/intflow/pig/pig_train_json/00236.json'
# # json_path = '/data2/intflow/cow/cow_train_json/00464.json'
# with open(json_path, 'r') as f:
#     anns = json.load(f)

# record = {}

# filename = json_path.replace('_json', '_image').replace('.json', '.jpg')
# obj_info = anns['ANNOTATION_INFO'][0]
# keypoints = obj_info['KEYPOINTS']
# keypoints = np.array(keypoints).reshape(-1, 3)

# img = cv2.imread(filename)
# for i, (x,y,_) in enumerate(keypoints):
#     cv2.circle(img, (x,y), 3, (255,0,0), -1)
#     cv2.putText(img, str(i), (x+5,y), 4, 1.0, (255,0,0))

# x_min, x_max = min(keypoints[:,0])-30, max(keypoints[:,0])+30
# y_min, y_max = min(keypoints[:,1])-300, max(keypoints[:,1])+30

# img_crop = img[y_min:y_max, x_min:x_max]
# plt.imshow(img_crop)
# plt.show()


# In[7]:


keypoint_names = {
    'cow': list(map(int, range(13))),
    'pig': list(map(int, range(8)))
}

keypoint_flip_map = {
    'cow': [('5', '7'), ('6', '8'), ('9', '11'), ('10', '12')],
    'pig': [('2', '3'), ('4','5'), ('6', '7')],
}


# In[8]:


class_to_id = {
    'cow': {
        'eating': 0,
        'standing': 1,
        'lying': 2,
        'sitting': 3,
        'tailing': 4,
        'head shaking': 5,
    },
    
    'pig': {
        'eating': 0,
        'standing': 1,
        'lying': 2,
        'sitting': 3,
    }
}


# In[9]:


import itertools
from detectron2.structures import BoxMode

def get_dataset_dicts(root_dir, task, d):
    assert task in ['pig', 'cow']
    assert d in ['train', 'val']
    json_files = sorted(glob(os.path.join(root_dir, task, '%s_train_json'%task, '*.json')))

    if d == 'train':
        json_files = json_files[int(len(json_files) * 0.05) :]
    else:
        json_files = json_files[:int(len(json_files) * 0.05)]
    
    dataset_dicts = []
    for idx, json_path in enumerate(json_files):
        try:
            with open(json_path, 'r') as f:
                anns = json.load(f)
            
            idx = int(idx)
            record = {}

            filename = json_path.replace('_json', '_image').replace('.json', '.jpg')
            height, width = anns['IMAGE']['HEIGHT'], anns['IMAGE']['WIDTH']

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
                
            objs = []
            for obj_info in anns['ANNOTATION_INFO']:
                category_id = class_to_id[task][obj_info['ACTION_NAME']]
                keypoints = obj_info['KEYPOINTS']
                
                x_min, x_max = min(keypoints[::3]), max(keypoints[::3])
                y_min, y_max = min(keypoints[1::3]), max(keypoints[1::3])

                x_min -= 30
                x_max += 30
                y_max += 30
                if task == 'pig':
                    y_min -= 200
                else:
                    y_min -= 30
                
                objs.append({
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "keypoints": obj_info['KEYPOINTS'],
                    "category_id": category_id
                })

            record["annotations"] = objs
            dataset_dicts.append(record)
            
        except Exception as e:
            print(filename, e)
            
    return dataset_dicts


# In[10]:




# In[12]:


for d in ["train", "val"]:
    DatasetCatalog.register("dataset_%s_%s" % (task, d), lambda d=d: get_dataset_dicts(root_dir, task, d))
    MetadataCatalog.get("dataset_%s_%s" % (task, d)).set(thing_classes=list(class_to_id[task].keys()),
                                            keypoint_names=keypoint_names[task],
                                            keypoint_flip_map=keypoint_flip_map[task])


# In[ ]:


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("dataset_%s_train" % task,)
cfg.DATASETS.TEST = ("dataset_%s_val" % task,)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = [60000,80000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(class_to_id[task].keys()))  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(keypoint_names[task])  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = 'output_%s' % task

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

