import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from encoder import Encoder 
from decoder import Decoder
from beam import BeamSearch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Seq2Seq(nn.Module):
    
    '''
    This class incorporate the whole transliteration model. It calls encoder and pass output of encoder
    to decoder with or wihout attention. Parameters are specified in constructor.
    '''
    
    def __init__(self, 
                 input_seq_length = 32,
                 output_seq_length = 29,
                 encoder_input_dimension = 29, 
                 decoder_input_dimension = 72,
                 encoder_hidden_dimension = 256, 
                 decoder_hidden_dimension =256,
                 encoder_embed_dimension = 256, 
                 decoder_embed_dimension = 256, 
                 bidirectional = True,
                 encoder_num_layers = 3,
                 decoder_num_layers = 3,
                 cell_type = 'lstm', 
                 dropout = 0.3,
                 beam_width = 3,
                 device = device,
                 attention = True
                ):
        
        
        super(Seq2Seq, self).__init__()
        self.detail_parameters = {
         'input_seq_length': input_seq_length,
         'output_seq_length' : output_seq_length,
         'encoder_input_dimension' : encoder_input_dimension, 
         'decoder_input_dimension' : decoder_input_dimension,
         'encoder_hidden_dimension' : encoder_hidden_dimension,
         'encoder_embed_dimension' : encoder_embed_dimension, 
         'decoder_hidden_dimension':decoder_hidden_dimension,
         'decoder_embed_dimension' : decoder_embed_dimension, 
         'bidirectional' : bidirectional,
         'encoder_num_layers' : encoder_num_layers,
         'decoder_num_layers' : decoder_num_layers,
         'cell_type' :cell_type, 
         'dropout' : dropout, 
         'device' : device.type
        }
        # Input sequence length => max_length of english
        self.input_seq_length = input_seq_length
        
        # Output sequence length => max_length of malayalam
        self.output_seq_length = output_seq_length
        
        # total number of english characters
        self.encoder_input_dimension = encoder_input_dimension
        
        # total number of malayalam characters
        self.decoder_input_dimension = decoder_input_dimension
        
        # Hidden dim for encoder
        self.encoder_hidden_dimension = encoder_hidden_dimension
        
        # Hidden dim for decoder
        self.decoder_hidden_dimension = decoder_hidden_dimension
        
        # Dimension to which we need to embed our source input
        self.encoder_embed_dimension = encoder_embed_dimension
        
        # Dimension to which we need to embed our target input
        self.decoder_embed_dimension = decoder_embed_dimension
        
        # Whether bidirection needed or not and sets its value as 2, so as to multiply hidden by 2
        self.direction = bidirectional
        self.direction_value = 2 if bidirectional else 1
        
        # Number of layers for encoder and decoder
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        
        # Which cell type to use
        self.cell_type = cell_type 
        
        # Whether to use dropout or not
        self.dropout = dropout
        self.device = device
        
        self.softmax = F.softmax
        
        # fix beam width
        self.beam_width = beam_width
        
        # Whether to use attention or not 
        self.use_attention = attention
        
        # Linear Weights so as to make encoder and decoder dimension same (i.e., if they differ by hidden dim or layer)
        self.enc_dec_linear1 = nn.Linear(encoder_hidden_dimension,decoder_hidden_dimension)
        self.enc_dec_linear2 = nn.Linear(encoder_num_layers*self.direction_value,decoder_num_layers*self.direction_value)
        
        # Linear Weights so as to make encoder and decoder cell's dimension same (i.e., if they differ by hidden dim or layer)
        self.enc_dec_cell_linear1 = nn.Linear(encoder_hidden_dimension,decoder_hidden_dimension)
        self.enc_dec_cell_linear2 = nn.Linear(encoder_num_layers*self.direction_value,decoder_num_layers*self.direction_value)
        
        # Linear Weights so as to make encoder and decoder attention dimension same (i.e., if they differ by hidden dim or layer)
        self.enc_dec_att_linear1 = nn.Linear(encoder_hidden_dimension,decoder_hidden_dimension)
        self.enc_dec_att_linear2 = nn.Linear(encoder_num_layers*self.direction_value,decoder_num_layers*self.direction_value)
        
        # initialize encoder
        self.encoder = Encoder(input_dimension = self.encoder_input_dimension,
                               embed_dimension = self.encoder_embed_dimension, 
                               hidden_dimension =  self.encoder_hidden_dimension,
                               cell_type = self.cell_type,
                               layers = self.encoder_num_layers,
                               bidirectional = self.direction,
                               dropout = self.dropout, 
                               device = self.device
                              )
        
        # initialize decoder
        self.decoder = Decoder(
                               input_dimension = self.decoder_input_dimension,
                               embed_dimension = self.decoder_embed_dimension,
                               hidden_dimension = self.decoder_hidden_dimension,
                               cell_type = self.cell_type,
                               layers = self.decoder_num_layers,
                               dropout = self.dropout, 
                               device = self.device,
                                use_attention = self.use_attention
                               )
        
    def getParams(self):
        return self.detail_parameters
    
    def forward(self, input, target ,teacher_force, acc_calculate = False):
        
        batch_size = input.shape[0]
        
        #initialize hidden dimension o pass to encoder
        enc_hidden = self.encoder.init_hidden(batch_size)
        
        # if lstm then initialize cell also
        if self.cell_type == 'lstm':
            cell = self.encoder.init_hidden(batch_size)
        else:
            cell = None
        
        encoder_outputs = None
        
        # if using attention, then encoder outputs should be stored 
        if self.use_attention:
            encoder_outputs = torch.zeros(self.input_seq_length,self.direction_value*self.decoder_num_layers,batch_size,self.decoder_hidden_dimension,device=device)
        
        # Pass input to encoder one by character in batch fashion
        for t in range(self.input_seq_length):
            enc_output,enc_hidden, cell = self.encoder.forward(input[:,t].unsqueeze(0), enc_hidden, cell)
            
            # Store encoder outputs, by first converting into same dimesnion by linear layers
            if self.use_attention:
                enc_hidden_new = enc_hidden
                enc_hidden_new = self.enc_dec_att_linear1(enc_hidden_new)
                enc_hidden_new = enc_hidden_new.permute(2,1,0).contiguous()
                enc_hidden_new = self.enc_dec_att_linear2(enc_hidden_new)
                enc_hidden_new = enc_hidden_new.permute(2,1,0).contiguous()
                encoder_outputs[t] = enc_hidden_new
        
        # Encoder's last state is decoders first state
        enc_last_state = enc_hidden
        
        # predicted to store all predictions by model to calculate loss
        predicted = torch.zeros(self.output_seq_length, batch_size, self.decoder_input_dimension,device = self.device)
        
        # Store all attention weights, so can be used for plotting attn heatmaps
        attn_weights = torch.zeros(self.output_seq_length, self.input_seq_length, self.direction_value*self.decoder_num_layers ,batch_size, device = self.device)
        
        # Encoders last state is decoders hidden also ransform in case they are of different dimension
        dec_hidden = enc_last_state
        dec_hidden = self.enc_dec_linear1(dec_hidden)

        dec_hidden = dec_hidden.permute(2,1,0).contiguous()
        dec_hidden = self.enc_dec_linear2(dec_hidden)
        dec_hidden = dec_hidden.permute(2,1,0).contiguous()
        
        # Here also, encoders last cell is decoders first cell, also transform to same dimesnion
        if  self.cell_type == 'lstm':
            cell = self.enc_dec_cell_linear1(cell)
            cell = cell.permute(2,1,0).contiguous()
            cell = self.enc_dec_cell_linear2(cell)
            cell = cell.permute(2,1,0).contiguous()
            

        # output at start is all 1's <SOS>
        output = torch.ones(1,batch_size,dtype=torch.long, device=self.device)
        predicted[0,:,1]=torch.ones(batch_size)
        attention_weights = None
        
        
        # Do decoding by char by char fashion by batch   
        for t in range(1,self.output_seq_length):
            # if teacher forcing, then pass target directly
            if teacher_force:
                output,dec_hidden,cell,attention_weights=self.decoder.forward(target[:,t-1].unsqueeze(0),dec_hidden,cell,encoder_outputs)
                predicted[t] = output.squeeze(0)

            else:
                # if beam is to be used, call beam instead of passing output from decoder
                if self.beam_width > 1 and acc_calculate:
                    beam = BeamSearch()
                    beam.beamSearch(self, output,dec_hidden,cell, predicted)
                    break
                    
                # call decoder one at a time
                output,dec_hidden,cell,attention_weights=self.decoder.forward(output,dec_hidden,cell,encoder_outputs)
                #store output in prediced (it containes probabilities)
                predicted[t] = output.squeeze(0)
                if self.use_attention:
                    attn_weights[t] = torch.sum(attention_weights, axis = 3)
                    
                # Convert output such that, it can be easily given to input
                output = self.softmax(output,dim=2)
                output = torch.argmax(output,dim=2)

        
        return predicted,attn_weights