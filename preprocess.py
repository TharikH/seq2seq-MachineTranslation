from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import torch
import numpy as np

#specify max length of sequence
mal_sequence_length = 29
eng_sequence_length = 32


# Function to load data from the data set and create and array corresponding to it
# Class to create dataset, so can be passed to pytorch dataloader
class MakeDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.int64)
        self.y = torch.tensor(y,dtype=torch.int64)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

  
    def __len__(self):
        return self.len


def loadData(PATH = "aksharantar/aksharantar_sampled/mal/mal_test.csv"):  
    arr = np.loadtxt(PATH,
                 delimiter=",", encoding = 'utf-8',dtype='str')
    num_sample = arr.shape[0]
    x,y = arr[:,0],arr[:,1]
    X = np.zeros((num_sample,eng_sequence_length)) # input
    Y = np.zeros((num_sample,mal_sequence_length)) # target

    for i in range(num_sample):

        X[i][0] = 1
        Y[i][0] = 1
        for j in range(len(x[i])):
            if(english_dict.get(x[i][j]) != None):
                X[i][j+1] = english_dict[x[i][j]]
            else:
                X[i][j+1] = 0

        X[i][len(x[i])+1]=2

        for j in range(len(y[i])):
            if(malayalam_dict.get(y[i][j]) != None):
                Y[i][j+1] = malayalam_dict[y[i][j]]
            else:
                Y[i][j+1] = 0

        Y[i][len(y[i])+1] = 2
        
    return X, Y



# Load Data to capture all characters
arr = np.loadtxt("aksharantar_sampled/mal/mal_train.csv",
                 delimiter=",", encoding = 'utf-8',dtype=str)


num_sample = arr.shape[0]
x_train,y_train = arr[:,0],arr[:,1]
english_index = 3
mal_index = 3
english_dict = {}
malayalam_dict = {}
english_index_dict = {}
malayalam_index_dict = {}

# Create dictionary for malayalam and english

for i in range(num_sample):
    for j in range(len(x_train[i])):
        
        if(english_dict.get(x_train[i][j]) == None):
            english_dict[x_train[i][j]]=english_index
            english_index_dict[english_index] = x_train[i][j]
            english_index+=1
        
    for j in range(len(y_train[i])):
            
        if(malayalam_dict.get(y_train[i][j]) == None):
            malayalam_dict[y_train[i][j]]=mal_index
            malayalam_index_dict[mal_index] = y_train[i][j]
            mal_index+=1

# Adding start, stop and padding symbols
malayalam_index_dict[1] = '<S>'
english_index_dict[1] = '<S>'

malayalam_index_dict[2] = '<E>'
english_index_dict[2] = '<E>'

malayalam_index_dict[0] = '<P>'
english_index_dict[0] = '<P>'
