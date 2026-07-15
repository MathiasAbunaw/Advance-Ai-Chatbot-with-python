import os #check if something exist 
import json #gain access to the json file
import random #for the user random responds 

import nltk #allow us to split the user text into tokens  focus on 
import numpy as np

#uses to defind our narual network and training it
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#nltk.download('punkt_tab') #This is needed for every nltk

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

        return x
class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        #Both documents and vocabulary is what will be used inorder to turn out sentances into values
        self.documents = [] 
        self.vocabulary = []


        self.intents = [] #This is what we will use inorder to assign the probabilities
        self.intents_responses = [] #This will be the list of intented responsis the bot will give or select

        self.function_mappings = function_mappings

        self.X = None
        self.Y = None

    @staticmethod #Cant have a self
    def tokenize_and_lemmatize(text): #This will be used inorder to break down the text into their word stem. SO instead of having diff variants of the same word we will break it down to only contain the stem/origin of the word
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        #breaking down text into word and breaking down the word into their stem word and ignoring the the case types
        words = [lemmatizer.lemmatize(word.lower()) for word in words] 
        return words

    def bag_of_words(self, words): # Encode the word with 0 and 1 depending if they are apart of the vocabulary
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatiser = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f: #opens the json file
                intents_data = json.load(f) #loads the json file within intents_data variable

                self.intents = []
                self.intents_responses = {}

                for intent in intents_data['intents']: #This will load all and only the intents within the json file
                    if intent['tag'] not in self.intents: #If the intents tags is not within the intents list then we will add it
                        self.intents.append(intent['tag']) #This will add the tags within the intents list
                        self.intents_responses[intent['tag']] = intent['responses'] #This will add the responds within the intents_responses list based on the tags
                    for pattern in intent['patterns']:
                        pattern_words = self.tokenize_and_lemmatize(pattern)
                        self.vocabulary.extend(pattern_words)
                        self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self): #This will be used inorder to turn the values into zero and one inorder 0 for useless word and 1 for important word
        bags = []
        indices = []
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags)
        self.y = np.array(indices)
    def train_model(self, batch_size, lr, epochs): #Lr: is how quickly out model is going to move into the direction of steapest decent, Batch size: How many instants we are going to parelel, Epochs: How many we time we are going to see the same data. 
        X_tensor = torch.tensor(self.X, dtype= torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size= batch_size, shuffle= True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr= lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+ 1}: Loss: {running_loss/ len(loader):.4f}")
    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weight_only = True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype = torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim = 1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None
                

def get_stocks():
    stocks = ['APPL', 'Meta', 'NVDA', 'GS', 'MSFT']
    return random.sample(stocks, 3)

if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size = 8, lr = 0.001, epochs = 100)

    assistant.save_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input("Enter your message:")

        if message == "/quit":
            break

        print(assistant.process_message(message))
