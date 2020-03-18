# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 07/11/19 
# @Contact: michealabaho265@gmail.com

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath('../'))
import data_prep as dp
from seq_models.biLSTM import BiLstm_model
from seq_models.biGRU import BiGRU_model
from seq_models.crf import crf
import numpy as np
import helper_functions as utils
import pickle

class NNModel(nn.Module):
    def __init__(self, config):
        super(NNModel, self).__init__()
        neural = config.pop('neural')
        if 'crf' in config and config['crf'] == True:
            self.crf_layer = config['crf']
            if neural.lower() == 'lstm':
                self._encode = BiLstm_model(**config).to(dp.device)
            elif neural.lower() == 'gru':
                self._encode = BiGRU_model(**config).to(dp.device)
            # define crf_layer
            crf_tag_map = config['tag_map']
            last_index = list(crf_tag_map.values())[-1]
            for i in ['start_token', 'stop_token']:
                crf_tag_map[i] = last_index + 1
                last_index += 1
            self.crf_m = crf(target_size=len(crf_tag_map), tag_map=crf_tag_map).to(dp.device)
        else:
            self.crf_layer = False
            if neural.lower() == 'lstm':
                print('\n--------Bio-BiLSTM--------\n')
                self._encode = BiLstm_model(**config).to(dp.device)
            elif neural.lower() == 'gru':
                print('\n--------Bio-GRU--------\n')
                self._encode = BiGRU_model(**config).to(dp.device)

        self.hidden = self._encode.init_hidden()

    def forward(self, input_seq, pos_seq, hidden, targets):
        features = self._encode(input_seq, pos_seq, hidden)
        if self.crf_layer:
            prediction_score = self.crf_m.forward(features)
            actual_score = self.crf_m.calculate_gold_score(features, targets.view(targets.size(0)))
            return prediction_score - actual_score, features
        class_scores = F.log_softmax(features, dim=1)
        return class_scores

    def decode(self, feats):
        return self.crf_m.viterbi_decode(feats)