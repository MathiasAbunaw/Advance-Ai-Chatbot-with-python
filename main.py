import os #check if something exist 
import json #gain access to the json file
import random #for the user random responds 

import nltk #allow us to split the user text into tokens  focus on 
import numpy as np

#uses to defind our narual network and training it
import torch
import torch.nn as nntoken
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

nltk.download('punkt_tab') #This is needed for every nltk

class ChatbotModel(nn.Module): #nn.module mean my class is inheriting PyTorch build in nn.Module class. BC PyTorch already build a neural network called nn.Module and by including it as a paremeter my chatbot starts off with everything that blueprint has
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        #my Input will flow through through all three layers and come out the other side as a prediction of which intent the user's message belongs to 
        self.fc1 = nn.Linear(input_size, 128) #Layer 1: takes my input and outputs 128 values
        self.fc2 = nn.Linear(128, 64) #Layer 2: Takes 128 values, output 64
        self.fc3 = nn.Linear(64,output_size) #Layer 3: Takes 64 values output final prediction 