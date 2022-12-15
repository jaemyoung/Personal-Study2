# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:23:47 2022

@author: user
"""

import math
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

import pickle
with open('data/sorted_library_1448.pkl', 'rb') as f: 
	node_feature = pickle.load(f) 
with open('data/have_requirelist_data_1971.pkl', 'rb') as f: 
	repo_list = pickle.load(f)  
    
model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = model.encode("proton exchange membrane")
s2 = model.encode("hybrid car")
cos_sim(s1,s2)

# description에서 keyphrase를 뽑고 해당 keyphrase를 fine tuning시 학
#%%
import numpy as np
from numpy import dot
from numpy.linalg import norm
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))