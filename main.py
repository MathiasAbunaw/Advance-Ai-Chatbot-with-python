import os #check if something exist 
import json #gain access to the json file
import random #for the user random responds 

import nltk #allow us to split the user text into tokens  focus on 
import numpy as np

#uses to defind our narual network and training it
import torch
import torch.nn as nn
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

        self.relu = nn.ReLU() #If the value if negative it replaces with 0. In doing so a negative value meansnthis signal isn't useful, ignore it. a positive value means this matters, pass it forward it a way to understand which values are useful and which are not 
        self.dropout = nn.Dropout(0.5) #Randomly switches of 50% of the neuron during training so the machine does not just memorize the training data set so that multiple nearons can handle it 

    def forward(self, x): #This is how we will get the output
        x = self.relu(self.fc1(x))
        x= self.dropout(x)
        x = self.relu(self.fc2(x))
        x= self.dropout(x)
        x = self.fc3(x)

class ChatbotAssistant:
    def __init__(self, intents_path, function_mapping = None):
        self.model = None
        self.intents_path = intents_path

        #Both documents and vocabulary is what will be used inorder to turn out sentances into values
        self.documents = [] 
        self.vocabulary = []


        self.intents = [] #This is what we will use inorder to assign the probabilities
        self.intents_responses = [] #This will be the list of intented responsis the bot will give or select

        self.function_mapping = function_mapping

        self.X = None
        self.Y = None

    @staticmethod
    def tokenize_and_lemmatize(text): #This will be used inorder to break down the text into their word stem. SO instead of having diff variants of the same word we will break it down to only contain the stem/origin of the word
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        #breaking down text into word and breaking down the word into their stem word and ignoring the the case types
        words = [lemmatizer.lemmatize(word.lower()) for word in words] 
        return words

    @staticmethod
    def bag_of_words(words, vocabulary ): # Encode the word with 0 and 1 depending if they are apart of the vocabulary
        return [1 if word in words else 0 for word in vocabulary]

    def parse_intents(self):
        lemmatiser = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

                intents = []
                intents_responses = []
                documents = []

                for intent in intents