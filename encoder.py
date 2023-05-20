import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    '''
     Encoder network in which hidden size, input dimension, embedding dimension, etc 
     can be specified for training.
    '''
    def __init__(self,
                    input_dimension = 72,
                    embed_dimension = 64,
                    hidden_dimension = 256,
                    cell_type = 'gru',
                    layers = 2,
                    bidirectional = True,
                    dropout = 0,
                    device = device
                ):
        
        super(Encoder, self).__init__()

        self.detail_parameters = {
            'input_dimension' : input_dimension,
            'embed_dimension' : embed_dimension,
            'hidden_dimension' : hidden_dimension,
            'cell_type' : cell_type,
            'dropout' : dropout,
            'layers' : layers,
            'direction_value' : 2 if bidirectional else 1,
            'device' : device.type,
        }
        
        # total number of english characters
        self.input_dimension = input_dimension
        
        # Dimension to which we need to embed our source
        self.embed_dimension = embed_dimension
        
        # Embedding the input
        self.embedding = nn.Embedding(self.input_dimension, self.embed_dimension)
        
        
        
        # Number of neurons in hidden layers
        self.hidden_dimension = hidden_dimension
        
        # Which type to use - RNN, GRU, LSTM
        self.cell_type = cell_type
        
        # Number of layers for hidden 
        self.layers = layers
        
        
        # Dropout to add onto embedded input
        self.dropout = nn.Dropout(dropout)
        
        # If bidirection then hidden size must be doubled.
        if bidirectional :
            self.direction_value = 2 
        else :
            self.direction_value = 1
        
        #device to use gpu or cpu
        self.device = device

        

        if self.cell_type == 'rnn':
            # type of cell to use is rnn
            self.encoder_type = nn.RNN(
                          input_size= self.embed_dimension,
                          num_layers= self.layers,
                          hidden_size= self.hidden_dimension,
                          dropout = dropout,
                          bidirectional= bidirectional)
            
        elif self.cell_type == "gru":
            # type of cell to use is gru
            self.encoder_type = nn.GRU(
                          input_size= self.embed_dimension,
                          num_layers= self.layers,
                          hidden_size= self.hidden_dimension,
                          dropout = dropout,
                          bidirectional= bidirectional)
        elif self.cell_type == "lstm":
            # type of cell to use is lstm
            self.encoder_type = nn.LSTM(
                          input_size= self.embed_dimension,
                          num_layers= self.layers,
                          hidden_size= self.hidden_dimension,
                          dropout = dropout,
                          bidirectional= bidirectional)
            
            
    def forward(self, input, hidden, cell=None):
        
        # First convert sequence to embedding
        embedded = self.embedding(input)
        
        # Apply dropout also
        embedded = self.dropout(embedded)
        
        #Then choose type of rnn to run using pytorch 
        if self.cell_type == 'lstm':
            output,(hidden,cell) = self.encoder_type(embedded, (hidden,cell))
        else:
            output, hidden = self.encoder_type(embedded, hidden)

        return output, hidden, cell

    def getParams(self):
        return self.detail_parameters
    
    def init_hidden(self, batch):
        # Initialize the hidden state to zeros
        return torch.zeros(self.direction_value*self.layers,batch,self.hidden_dimension,device=device)