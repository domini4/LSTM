# -*- coding: utf-8 -*-
"""
CPSC 8430: HW2 Video image captioning

@author: James Dominic

Train the RNN network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from dict import dict
from model import modelsRNN, encoder, decoder
from dataset import training_data


# Create mini-batch tensors from the list of tuples (image, caption)
def minibatch(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    training_json = 'MLDS_hw2_1_data/training_label.json'
    training_feats = 'MLDS_hw2_1_data/training_data/feat'
    testing_json = 'MLDS_hw2_1_data/testing_label.json'
    testing_feats = 'MLDS_hw2_1_data/testing_data/feat'
    
    helper = dict(training_json, 3)
    train_dataset = training_data(label_json=training_json, training_data_path=training_feats, helper=helper, load_into_ram=True)
    test_dataset = training_data(label_json=testing_json, training_data_path=testing_feats, helper=helper, load_into_ram=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, collate_fn=minibatch)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=minibatch)
    
    
    word_size = 1024
    input_size = 4096
    hidden_size = 512
    dropout_percentage = 0.1
    output_size = helper.vocab_size
    
    epochs = 200
    ModelSaveLoc = 'SavedModel'
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)
    
    encoder = encoder(input_size, hidden_size, dropout_percentage)
    decoder = decoder(hidden_size, output_size, output_size, word_size, dropout_percentage)
    model = modelsRNN(encoder, decoder)
    
    
    # train the network
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    loss_new = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(1, epochs+1):
        for idx, (feat, truth, lengths) in enumerate(train_dataloader):
            feat, truth = feat.to(device), truth.to(device)
            optimizer.zero_grad()
            prob, pred = model(feat, target_sentences=truth)
            truth = truth[:, 1:]  
            
            # calculate loss
            batch_size = len(prob)
            prediction = None
            ground = None        
            for batch in range(batch_size):
                predict = prob[batch]
                ground_truth = truth[batch]
                length = lengths[batch] -1
    
                predict = predict[:length]
                ground_truth = ground_truth[:length]
                
                if prediction == None:
                    prediction = predict
                    ground = ground_truth
                else:
                    prediction = torch.cat((prediction, predict), dim=0)
                    ground = torch.cat((ground, ground_truth), dim=0)
            
            loss = loss_new(prediction, ground)
            loss.backward()
            optimizer.step()
            
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch, epochs, loss.item()))
        
    # save model
    torch.save(model, "{}/{}.h5".format(ModelSaveLoc, 'model'))