# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 31/10/19 
# @Contact: michealabaho265@gmail.com

import torch
import torch.nn as nn


class crf(nn.Module):
    def __init__(self, target_size, tag_map):
        super(crf, self).__init__()
        self.target_size = target_size
        self.tag_map = tag_map
        #defining a default set of transition probabilities
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size))
        self.transitions[tag_map['start_token'], :] = -10000
        self.transitions[:, tag_map['end_token']] = -10000

    def forward(self, input_features, target_states):
        #initialize viterbi variables
        init_transitions = torch.full((1, self.target_size), -10000.)
        init_transitions[0][self.index2tag['start_token']] = 0
        forward_vars = init_transitions

        for f in input_features:
            alphas_t = []
            for tag in range(self.target_size):
                emit_score = input_features[tag].view(1,-1).expand(1, self.target_size)
                trans_score = self.transitions[tag].view(1,-1)
                next_tag_var = forward_vars + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_vars = torch.cat(alphas_t).view(1,-1)
        terminal_var = forward_vars + self.transitions[self.tag_map['stop_token']]
        alpha = log_sum_exp(terminal_var)
        neg_log_likelihood = self.calculate_gold_score(target_states) - alpha
        return neg_log_likelihood

    def calculate_gold_score(self, target):
        score = torch.zeros(1)
        torch.cat([torch.tensor(self.tag_map['start_token']), target])
        for i,f in enumerate(self.features):
            score = score + self.transitions[target[i+1], target[i]] + f[i+1]
        score = score + self.transitions[self.tag_map['stop_token'], target[-1]]
        return score

    def viterbi_decode(self, input_features):
        back_pointers = []
        init_transitions = torch.full((1, self.target_size), -10000.)
        init_transitions[0][self.index2tag['start_token']] = 0
        forward_vars = init_transitions

        for f in input_features:
            best_path = []
            decoded_scores = []
            for tag in range(self.target_size):
                next_tag_var = forward_vars + self.transitions[tag]
                best_tag, best_tag_loc = torch.max(next_tag_var, 1)
                best_path.append(best_tag_loc)
                decoded_scores.append(best_tag)

            forward_vars = (torch.cat(decoded_scores) + f).view(1,-1)
            back_pointers.append(best_path)

        #transitioning to the stop token
        terminal_var = forward_vars + self.transitions[self.tag_map['stop_token']]
        path_score, best_tag_loc = torch.max(terminal_var, 1)

        #follow the back pointers to decode the best path
        decoded_best_path = [best_tag_loc]
        for v in reversed(back_pointers):
            best_tag_loc = v[best_tag_loc]
            decoded_best_path.append(best_tag_loc)

        start = decoded_best_path.pop()
        assert start == self.tag_map['start_token']
        decoded_best_path.reverse()
        return path_score, decoded_best_path

def log_sum_exp(vec):
    max_score = torch.max(vec)
    max_score_broad_cast = max_score.view(1,-1).expand(1, vec.size(1))
    log_sum = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broad_cast)))
    return log_sum
