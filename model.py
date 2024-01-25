#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPSC 8430: HW2 Video image captioning

@author: James Dominic

Define RNN models
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np


class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_percentage):
        super(encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_percentage)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        batch_size, seq_length, feat_n = x.size()    
        x = x.view(-1, feat_n)
        embedding_size = self.dropout(self.embedding(x))
        rnn_input = embedding_size.view(batch_size, seq_length, self.hidden_size)    
        output, (hidden_state, cell) = self.lstm(rnn_input)

        return output, hidden_state, cell
        


class decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage):
        super(decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.energy1 = nn.Linear(hidden_size*2, hidden_size)
        self.energy2 = nn.Linear(hidden_size, hidden_size)
        self.energy3 = nn.Linear(hidden_size, 1)
        
        self.lstm = nn.LSTM(hidden_size+word_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, last_hidden_state, encoder_output, cell_state, device, targets=None):
        _, batch_size, _ = last_hidden_state.size()
        if(last_hidden_state is None):
            hidden_state = None
        else:
            hidden_state = last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long().to(device)
        prob = []
        pred = []
        

        # schedule sampling
        if self.training:
            targets = self.embedding(targets)
            _, seq_length, _ = targets.size()
        else:
            seq_length = 28

        for i in range(seq_length-1):
            if self.training:
                threshold = 0.7
                if random.uniform(0.05, 0.995) > threshold:
                    current_input_word = targets[:, i]
                else: 
                    current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            else:
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
                
                
            current_input_word = self.dropout(current_input_word)
            # compute attention
            batch_size, sequence_length, feat_n = encoder_output.size()
            hidden_reshaped = last_hidden_state.view(batch_size, 1, feat_n).repeat(1, sequence_length, 1)
            energy = self.energy1(torch.cat((hidden_reshaped, encoder_output), dim=2))
            energy = self.energy2(energy)
            energy = F.relu(self.energy3(energy))
            attention = F.softmax(energy, dim=0)
            x, y, z = attention.size()
            attention = attention.view(x, z, y)
            encoder_state = encoder_output.permute(0,1,2)
            context_vector = torch.bmm(attention, encoder_state).squeeze(1)
            
            # run rnn
            rnn_input = torch.cat((context_vector, current_input_word), dim=1).unsqueeze(1)
            rnn_out, (hidden_state, cell) = self.lstm(rnn_input, (hidden_state, cell_state))
            logprob = self.fc(rnn_out.squeeze(1))
            prob.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        prob = torch.cat(prob, dim=1)
        pred = prob.max(2)[1]

        return prob, pred


class modelsRNN(nn.Module):
    def __init__(self, encoder, decoder):
        super(modelsRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, avi_feats, target_sentences=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_outputs, last_hidden_state, cell_state = self.encoder(avi_feats)
        prob, pred = self.decoder(last_hidden_state, encoder_outputs, cell_state, device, targets=target_sentences)
        return prob, pred




















