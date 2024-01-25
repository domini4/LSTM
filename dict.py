# -*- coding: utf-8 -*-
"""
CPSC 8430: HW2 Video image captioning

@author: James Dominic

Create dictionary of words
"""
import json
import re


class dict():
    def __init__(self, dataset, min_words):
        # file path for training data
        self.file_labels = dataset
        
        self.min_words = min_words
        
        self.vocab_size = None
        self.itos = None
        self.stoi = None
        self._word_count = {}
        
        self.create_dic()
    
    def create_dic(self):
        with open(self.file_labels, 'r') as f:
            file = json.load(f)
    
        for vid in file:
            # for each video caption
            for sen in vid['caption']:
                # use regex to remove all punctuations and split words
                sentence = re.sub('[.!,;?]', ' ', sen).split()
                
                # maintain a count of repeating words
                for word in sentence:
                    self._word_count[word] = self._word_count.get(word, 0) + 1
    
        vocab = [k for k, v in self._word_count.items() if v > self.min_words]
            
        tokens = [('<PAD>', 0), ('<BOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
        self.itos = {i + len(tokens): w for i, w in enumerate(vocab)}
        self.stoi = {w: i + len(tokens) for i, w in enumerate(vocab)}
        for token, index in tokens:
            self.itos[index] = token
            self.stoi[token] = index
    
        self.vocab_size = len(self.itos) + len(tokens)
        
    def reannotate(self, sentence):
        sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        sentence = ['<BOS>'] + [w if (self._word_count.get(w, 0) > self.min_words) \
                                    else '<UNK>' for w in sentence] + ['<EOS>']
        return sentence

    def sentence2index(self, sentence):
        return [self.stoi[w] for w in sentence]
    def index2sentence(self, index_seq):
        return [self.itos[int(i)] for i in index_seq]