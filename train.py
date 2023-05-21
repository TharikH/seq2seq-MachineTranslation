import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 

from tqdm import tqdm
import heapq
import csv

import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import wandb

from seqtoseq import Seq2Seq 
from preprocess import *
import arguments
from beam import BeamSearch
from encoder import Encoder 
from decoder import Decoder 
from utils import *
import arguments

args = arguments.parsArg()

# Instantiates the device to be used as GPU/CPU based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

key = 'c425b887e2c725018a7f3a772582610fa54ef52c'
english_vocab_size = 29
malayalam_vocab_size = 72

# Load all data with specified path
X_train,y_train = loadData(PATH = "aksharantar_sampled/mal/mal_train.csv")
X_val,y_val = loadData(PATH = "aksharantar_sampled/mal/mal_valid.csv")
X_test,y_test = loadData(PATH = "aksharantar_sampled/mal/mal_test.csv")


# Create dataset to pass to dataloader

train_dataset = MakeDataset(X_train,y_train)
val_dataset = MakeDataset(X_val, y_val)
test_dataset = MakeDataset(X_test, y_test)


# Create dataloader so getting data in epochs is easy
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size)
val_loader = DataLoader(val_dataset,shuffle=True,batch_size=args.batch_size)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=args.batch_size)

model = Seq2Seq(
    input_seq_length = eng_sequence_length,
    output_seq_length = mal_sequence_length,
    encoder_input_dimension = english_vocab_size, 
    decoder_input_dimension = malayalam_vocab_size,

    encoder_hidden_dimension = args.encoder_hidden_size, 
    decoder_hidden_dimension = args.decoder_hidden_size,
    encoder_embed_dimension = args.encoder_embedding, 
    decoder_embed_dimension = args.decoder_embedding, 
    bidirectional = args.bidirectional,
    encoder_num_layers = args.num_encoder_layers,
    decoder_num_layers = args.num_decoder_layers,
    cell_type = args.cell_type, 
    dropout = args.dropout,
    beam_width = args.beam_width,
    device = device,
    attention = args.use_attention
)

beam = args.beam_width > 1
model.to(device)
epochs = args.epochs
wandb.login(key = key)
wandb.init(project = args.wandb_project,entity=args.wandb_entity)
wandb.run.name = f'epoch_{args.epochs}_enc_hs_{args.encoder_hidden_size}_dec_hs_{args.decoder_hidden_size}_enc_emb_{args.encoder_embedding}_dec_emb_{args.decoder_embedding}_bi_{args.bidirectional}_encl_{args.num_encoder_layers}_decl_{args.num_decoder_layers}_type_{args.cell_type}_drop_{args.dropout}_beam_{args.beam_width}_att_{args.use_attention}'

train_loss_list,val_loss_list,train_accuracy_list,val_accuracy_list = runModel(model, train_loader, val_loader, epochs, beam)


for i in range(epochs):
        wandb.log({'validation_loss': val_loss_list[i],
                  'training_loss': train_loss_list[i],
                  'validation_accuracy': val_accuracy_list[i],
                  'training_accuracy': train_accuracy_list[i]
                  })

test_loss,test_accuracy = testAccuracy(model, test_loader)
wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
          })

# Get test examples, so we can plot the attention part - heatmap
# test_loader = DataLoader(test_dataset,shuffle=True,batch_size=256)
test_input, test_labels = next(iter(test_loader))
model.eval()
# _weights will have attention weights
test_output,_weights = model.forward(test_input.to(device), None,False)

if model.use_attention:
    acc_output = F.softmax(test_output,dim=2)
    acc_output = torch.argmax(acc_output,dim=2)
    acc_output.shape
    acc_output = acc_output.T

    w =  torch.mean(_weights,axis=2)
    
    image = plotHeatMap(test_input,acc_output,w,english_index_dict, malayalam_index_dict)
    wandb.log({"attention_weights_matplotlib" : [wandb.Image(image,caption="Attention weights_matplotlib")]})
else:
    print("No Attention => No heatmap")

wandb.finish()
