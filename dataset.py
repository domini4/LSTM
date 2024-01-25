# -*- coding: utf-8 -*-
"""
CPSC 8430: HW2 Video image captioning

@author: James Dominic

Create dataset
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class training_data(Dataset):
    def __init__(self, label_json, training_data_path, helper, load_into_ram=False):
        # check if file path exists
        if not os.path.exists(label_json):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(label_json, __name__))
        if not os.path.exists(training_data_path):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(training_data_path, __name__))


        self.training_data_path = training_data_path
        # format (avi id, corresponding sentence)
        self.data_pair = []
        self.load_into_ram = load_into_ram
        self.helper = helper


        with open(label_json, 'r') as f:
            label = json.load(f)
        for d in label:
            for s in d['caption']:
                s = self.helper.reannotate(s)
                s = self.helper.sentence2index(s)
                self.data_pair.append((d['id'], s))

        if load_into_ram:
            self.avi = {}

            files = os.listdir(training_data_path)

            for file in files:
                key = file.split('.npy')[0]
                value = np.load(os.path.join(training_data_path, file))
                self.avi[key] = value


    def __len__(self):
        return len(self.data_pair)


    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        avi_file_path = os.path.join(self.training_data_path, '{}.npy'.format(avi_file_name))
        data = torch.Tensor(self.avi[avi_file_name]) if self.load_into_ram else torch.Tensor(np.load(avi_file_path))
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)
    

class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())

        return self.avi[idx]
    
