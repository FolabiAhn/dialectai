import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *




class EncoderRNN(nn.Module):
    def __init__(self, device, batch_size = 10, hidden_size=256):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size # For the moment    

        self.dropout_1 = nn.Dropout(0.2)
        
        self.gru_1 = nn.GRU(39, hidden_size, batch_first=True, num_layers=1, bidirectional=True)


    def forward(self, wave, hidden):

        # Started with the gru
        output, hidden = self.gru_1(wave, hidden)
        _, dim_1, dim_2 = output.shape
        # Separate the forward pass ----> batch, seq_len, num_directions, hidden_size
        forward = output.view(self.batch_size, dim_1, 2, self.hidden_size)[:, :, 0, :]
        # Separate the backward pass  
        backward = output.view(self.batch_size, dim_1, 2, self.hidden_size)[:, :, 1, :]
        # Sum the forward pass and the backward to form the output
        output = (forward + backward) 
        output = F.relu(output)
        # Pass through the dropout layer
        output = self.dropout_1(output)
        # I don't know what i will do with but i collect the last state for the moment
        hidden = hidden.view(1, 2,  self.batch_size, self.hidden_size)
        #[-1,:,:].unsqueeze(0).to(self.device)
        return output.squeeze(0), hidden 

    def initialize_hidden_state(self):
        return torch.zeros(2, self.batch_size, self.hidden_size, device=self.device)
    

    
    
class ConvBase(nn.Module):
    def __init__(self, hidden_size):
        super().__init__() 
        self.hidden_size = hidden_size
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=3, stride=(2,3))
        self.batchnorm2d_1 = nn.BatchNorm2d(hidden_size)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, dilation=1)
        self.conv2d_2 = nn.Conv2d(in_channels=256, out_channels=self.hidden_size, kernel_size=3, 
                                  stride=1, dilation=1)
    
    def forward(self, mfccs):
        # Started with the convolutionnal base
        conv_layer = self.conv2d_1(mfccs)
        conv_layer = self.conv2d_2(F.relu(conv_layer))
        conv_layer = self.batchnorm2d_1(F.relu(conv_layer))
        avg_layer = self.avg_pool_1(conv_layer)        
        # Reshape from (batch, channels, features, time) ----> (batch, time, channels * features)
        dim_0, dim_1, dim_2, dim_3 = avg_layer.shape
        output = avg_layer.reshape(dim_0, dim_3, dim_1 * dim_2)
        return output
    
    
class RnnBase(nn.Module):
    def __init__(self, device, hidden_size, batch_size, bn_in_feat, gru_in_feat):
        super().__init__() 
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size 
        self.dropout = nn.Dropout(0.2)
        self.batchnorm1d = nn.BatchNorm1d(bn_in_feat)
        self.gru = nn.GRU(gru_in_feat, self.hidden_size, batch_first=True, num_layers=1, bidirectional=True)
    
    
    def forward(self, seq, hidden):
        #hidden = self.initialize_hidden_state()
        # Beginning rnn bloc
        output, hidden = self.gru(seq, hidden)
        _, dim_1, dim_2 = output.shape
        # Separate the forward pass ----> batch, seq_len, num_directions, hidden_size
        forward = output.view(-1, dim_1, 2, self.hidden_size)[:, :, 0, :]
        # Separate the backward pass  
        backward = output.view(-1, dim_1, 2, self.hidden_size)[:, :, 1, :]
        # Sum the forward pass and the backward to form the output
        output = forward + backward
        output = F.relu(output)
        output = self.batchnorm1d(output)
        # Pass through the dropout layer
        output = self.dropout(output)
        # I don't know what i will do with but i collect the last state for the moment
        #self.last_state = hidden.view(2,  self.batch_size, self.hidden_size)[-1,:,:].unsqueeze(0).to(self.device)
        hidden = hidden.view(1, 2,  -1, self.hidden_size)
        h_forward = hidden[:, 0, :, :]
        h_backward = hidden[:, 1, :, :]
        # Type of merge chosen 
        hidden = h_forward + h_backward # Batch norm pour le hidden ?
        
        
        return output, hidden

        
    
class EncoderCONV2DRNN(nn.Module):
    def __init__(self, device, batch_size = 10, hidden_size=256):
        super().__init__() 
        self.batch_size = batch_size
        self.conv_base = ConvBase(hidden_size)
        self.hidden_size = hidden_size
        self.device = device
        self.rnn_base_1 = RnnBase(device, hidden_size, batch_size, bn_in_feat=198, gru_in_feat=1088)
        self.rnn_base_2 = RnnBase(device, hidden_size, batch_size, bn_in_feat=198, gru_in_feat=hidden_size)
        self.rnn_base_3 = RnnBase(device, hidden_size, batch_size, bn_in_feat=198, gru_in_feat=hidden_size)
        self.rnn_base_4 = RnnBase(device, hidden_size, batch_size, bn_in_feat=198, gru_in_feat=hidden_size)
        self.rnn_base_5 = RnnBase(device, hidden_size, batch_size, bn_in_feat=198, gru_in_feat=hidden_size)
        self.rnn_base_6 = RnnBase(device, hidden_size, batch_size, bn_in_feat=198, gru_in_feat=hidden_size)


    def forward(self, mfccs, hidden):
        
        #print("mfccs ----",mfccs.shape)
        # Convolutionnal base
        output = self.conv_base(mfccs)
        # Sequential bloc
        output, _ = self.rnn_base_1(output, hidden)
        copy_output = output.clone()
        
        output, _ = self.rnn_base_2(output, hidden)
        output = output + copy_output
        copy_output = output.clone()
        
        output, _ = self.rnn_base_3(output, hidden)
        output = output + copy_output
        copy_output = output.clone()
        
        output, _ = self.rnn_base_4(output, hidden)
        output = output + copy_output
        copy_output = output.clone()
        
        output, _ = self.rnn_base_5(output, hidden)
        output = output + copy_output
        copy_output = output.clone()

        output, hidden = self.rnn_base_6(output, hidden)
        output = output + copy_output
        
        #print("enc out 1----", output.shape)
        #print("enc out 2----", output.squeeze(0).shape)
        return output, hidden #output.squeeze(0), hidden  

    def initialize_hidden_state(self):
        return torch.zeros(2, self.batch_size, self.hidden_size, device=self.device)
    
    
class DecoderATTRNN1(nn.Module):
    """Bahdanau Baseline decoder"""
    
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, hidden_size, method='bahdanau_basic'):
        
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_1 = nn.GRU(embedding_dim + hidden_size, self.dec_units, batch_first=True)
        self.gru_2 = nn.GRU(self.dec_units, self.dec_units, batch_first=True)
        self.gru_3 = nn.GRU(self.dec_units, self.dec_units, batch_first=True)
        self.gru_4 = nn.GRU(self.dec_units, self.dec_units, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        # used for attention
        if method == 'bahdanau_basic':
            self.attention = BahdanauAttentionBase(units=dec_units, hidden_size=hidden_size)
        elif method == 'bahdanau_audio':
            self.attention = BahdanauAttentionAudio(kernel_size=2, units=dec_units, hidden_size=hidden_size)
        elif method == 'luong_dot':
            self.attention = LuongAttentionDot()
        elif method == 'luong_concat':
            self.attention = LuongAttentionConcat(units=dec_units, hidden_size=hidden_size)
        elif method == 'luong_general':
            self.attention = LuongAttentionGeneral(hidden_size=hidden_size)
        elif method == 'super_head':
            self.attention = SuperHeadAttention(units=dec_units, hidden_size=hidden_size)
        else:
            print(" ---- YOU SHOULD SPECIFY THE METHOD FOR THE ATTENTION MECHANISM !!! ---- ")
        
    
    def forward(self, input, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights, _ = self.attention(hidden, enc_output)
        # x shape after passing input through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(input)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        context_vector = torch.unsqueeze(context_vector, 1)
        x = torch.cat((context_vector, x), 2)
        # passing the concatenated vector to the GRU
        output, _ = self.gru_1(x)
        copy_output = output.clone()
        
        output, _ = self.gru_2(output)
        output = output + copy_output
        copy_output = output.clone()

        output, _ = self.gru_3(output)
        output = output + copy_output
        copy_output = output.clone()
        
        output, state = self.gru_4(output)
        output = output + copy_output
        
        # output shape == (batch_size * 1, hidden_size)
        output = output.reshape(-1, output.shape[2])
        # output shape == (batch_size, vocab)
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)

        return output, state, attention_weights
        
