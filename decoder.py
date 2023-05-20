import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Decoder(nn.Module):
    '''
    Decoder to decode to malayalam word. It also contain different parameters
    which is specified in the contructor. 
    '''
    def __init__(self,
                input_dimension = 26,
                embed_dimension = 64,
                hidden_dimension = 256,
                cell_type = 'lstm',
                layers = 2,
                use_attention = False,
                dropout = 0,
                bidirectional = True,
                device = device
                 ):
        
        super(Decoder, self).__init__()
        
        self.detail_parameters = {
            'input_dimension' : input_dimension,
            'embed_dimension' : embed_dimension,
            'hidden_dimension' : hidden_dimension,
            'cell_type' : cell_type,
            'layers' : layers,
            'device' : device.type,
            'dropout' : dropout,
            'use_attention' : use_attention,
        }

        # total number of malayalam characters
        self.input_dimension = input_dimension
        
        # Dimension to which we need to embed our input
        self.embed_dimension = embed_dimension  
        
        # Embedding the input
        self.embedding = nn.Embedding(self.input_dimension, self.embed_dimension)
        
        # Real input size given to nn.Decoder including attention dimension
        self.input_size = embed_dimension
        
        # Number of neurons in hidden layers
        self.hidden_dimension = hidden_dimension
        
        # If attention is being used then set this variable
        self.use_attention = use_attention
        
        # After applying weight, what should be the dimension should be
        self.attention_out_dimension = 1
        
        # Which type to use - RNN, GRU, LSTM
        self.cell_type = cell_type
        
        # Number of layers for hidden
        self.layers = layers
        
        # device to use gpu or cpu
        self.device = device
        
        # Dropout to add onto embedded input
        self.dropout = nn.Dropout(dropout)
        
        
        
        if bidirectional :
            self.direction_value = 2 
        else :
            self.direction_value = 1
            
        # Weights to multiply output we get from decoder
        self.W1 = nn.Linear(self.hidden_dimension*self.direction_value, self.hidden_dimension)
        self.W2 = nn.Linear(self.hidden_dimension, self.input_dimension)
        
        self.softmax = F.softmax

        if self.use_attention:
            self.input_size += self.hidden_dimension
            
            # Initialize 3 weight matrices, so we can muliply for attention
            self.U = nn.Sequential(nn.Linear( self.hidden_dimension, self.hidden_dimension), nn.LeakyReLU())
            self.W = nn.Sequential(nn.Linear( self.hidden_dimension, self.hidden_dimension), nn.LeakyReLU())
            self.V = nn.Sequential(nn.Linear( self.hidden_dimension, self.attention_out_dimension), nn.LeakyReLU())
        
        

        if self.cell_type == 'rnn':
            # type of cell to use is rnn
            self.decoder_type = nn.RNN(input_size= self.input_size,
                            dropout = dropout,     
                            num_layers= self.layers,
                            hidden_size= self.hidden_dimension,
                            bidirectional= bidirectional
                                 )
        elif self.cell_type == 'gru':
            # type of cell to use is gru
            self.decoder_type = nn.GRU(input_size= self.input_size, # to concat attention_output
                            num_layers= self.layers,
                            hidden_size= self.hidden_dimension,
                            dropout = dropout,
                            bidirectional= bidirectional
                                 )
        elif self.cell_type == "lstm":
            # type of cell to use is lstm
            self.decoder_type = nn.LSTM(input_size= self.input_size,
                            num_layers= self.layers,
                            hidden_size= self.hidden_dimension,
                            dropout = dropout,
                            bidirectional= bidirectional
                                  )


    def getParams(self):
        return self.detail_parameters
    
    def applyAttention(self, hidden, enc_output):
        
        '''
        It uses attention mechanism to include encoders weights to decoder.
        '''
        encoder_transform  = self.W(enc_output)
        hidden_transform =  self.U(hidden)

        concat_transform = encoder_transform + hidden_transform
        score = torch.tanh(concat_transform)
        
        score = self.V(score)

        # This will be our probability distribution for the attention weights (alpha)
        attention_weights = torch.softmax(score, dim=0)
        
        # conext vector to be concatenated to input
        context_vector = attention_weights * enc_output

        
        # To make dimension correct after attention so we can concat with input of decoder
        normalized_context_vector = torch.sum(context_vector,dim=0)
        normalized_context_vector = torch.sum(normalized_context_vector,dim=0).unsqueeze(0)
        
        return normalized_context_vector,attention_weights
    
    def forward(self, input, hidden, cell=None,encoder_outputs=None):
#         Incorporate dropout in embedding.
        output = self.dropout(self.embedding(input))
    
        attention_weights = None
        
#         If we are using attention, then we need to concatenate the context vector, which we obtain from attention
        if self.use_attention:
            context,attention_weights = self.applyAttention(hidden, encoder_outputs)
            output = torch.cat((output,context),2)
        
        if self.cell_type == 'lstm':
            output,(hidden,cell) = self.decoder_type(output,(hidden,cell))
        else:
            output, hidden = self.decoder_type(output, hidden)
            
        # Apply linear layers onto output obtained from decoder, to make it input dimension 
        output = self.W1(output)
        output = self.W2(output)
        
        return output, hidden, cell, attention_weights

    