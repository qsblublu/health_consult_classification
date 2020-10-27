import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import health_consult_class_map


class TextDataset(Dataset):
    def __init__(self, f_data: str, f_dict: str):
        with open(f_dict, mode='r') as f:
            self.word_dict = json.load(f)
        self.max_sentence_length = self.word_dict['max_sentence_len']
        self.data = pd.read_csv(f_data)

    def sentence_to_tensor(self, idx: int):
        age = int(self.data.loc[idx, '岁数'])
        gender = int(self.data.loc[idx, '性别'])
        sentence = self.data.loc[idx, '健康咨询描述']

        feature_arr = np.array([age, gender])

        words = sentence.split(' ')
        sentence_arr = np.array([int(self.word_dict[word]) for word in words])
        rest_len = self.max_sentence_length - len(sentence_arr)

        zero_arr = np.zeros(rest_len, dtype=np.int32)

        return torch.from_numpy(np.concatenate((feature_arr, sentence_arr, zero_arr), axis=0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.sentence_to_tensor(idx)
        label = self.data.loc[idx, 'disease_analysis_class']
        label = torch.tensor(health_consult_class_map[label])
        sample = {'sentence': sentence, 'label': label}

        return sample
