#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPSC 8430: HW2 Video image captioning

@author: James Dominic

TEst code
"""
import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
from dict import dict
from dataset import test_data
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('SavedModel/model.h5').to(device)

dataset = test_data('{}/testing_data/feat'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True)
training_json = 'MLDS_hw2_1_data/training_label.json'
helper = dict(training_json, 3)

model.eval()
sentences = []
for idx, (ids, feat) in enumerate(testing_loader):
    feat = feat.to(device)
    
    prob, pred = model(feat.float())
    # substute UNK with a high probabiblity word
    result = [[x if x != '<UNK>' else 'the' for x in helper.index2sentence(s)] for s in pred]
    result = [' '.join(s).split('<EOS>')[0] for s in result]
    result_tuple = zip(ids, result)
    for r in result_tuple:
        sentences.append(r)

if not os.path.exists(sys.argv[2].split("/")[0]):
    os.makedirs(sys.argv[2].split("/")[0])

with open(sys.argv[2], 'w') as f:
    for id, s in sentences:
        f.write('{},{}\n'.format(id, s))

# Bleu Eval
test = json.load(open('{}/testing_label.json'.format(sys.argv[1]),'r'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))
