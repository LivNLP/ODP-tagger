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
from outcome_train import load_and_process
import numpy as np
import helper_functions as utils

torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        tokens, pos = sentence
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(tokens).view(len(tokens), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(tokens), self.hidden_dim*2)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def evaluate(model, test_data, data_dir, index2tag):
    P,T,probs = [],[],[]
    with torch.no_grad():
        with open(os.path.join(data_dir, 'decoded.out'), 'w') as d, open(os.path.join(data_dir, 'baseline_model_results.txt'), 'w') as h:
            acc, acc_count = 0, 0
            for sent, tags in test_data:
                score, predicted = model(sent)
                target = [i.item() for i in tags]
                target_sentence, predicted_sentence = utils.extract_predictions_from_model_output(target, predicted, index2tag)

                for u,v in zip(target_sentence.split(), predicted_sentence.split()):
                    d.write('{} {}'.format(u.strip(), v.strip()))
                    d.write('\n')
                d.write('\n')

                acc += utils.token_level_accuracy(target, predicted)
                acc_count += 1
                P.append(predicted_sentence.split())
                T.append(target_sentence.split())
            evaluation_accuracy = float(acc/acc_count)
            print('Evaluation accuracy {}'.format(evaluation_accuracy))
            P = [i for j in P for i in j]
            T = [i for j in T for i in j]
            classes = [v for k,v in index2tag.items()]
            F_measure, AP, fig, classification = utils.metrics_score(target=T, predicted=P, cls=classes, prfs=True, probs=probs)
            h.write('Classification Matrix \n {} \n'.format(classification))
            h.write('Average Precision: {} \n'.format(AP))
            d.close()
            h.close()
            print(classification)
    return evaluation_accuracy, classification, fig

if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 150
    HIDDEN_DIM = 250
    file_path = '../../pico-outcome-prediction/corrected_outcomes/BIO-data/stanford'
    train_data_file = os.path.join(file_path, 'train_stanford_ebm.bmes')

    line_pairs, word_map, outputs, index2word, tag_map, pos_map, input_words, output_tags, num_words, num_pos, max_len = load_and_process(train_data_file)
    index2tag = dict((v, k) for k, v in tag_map.items())
    last_index = list(tag_map.values())[-1]
    for i in [START_TAG, STOP_TAG]:
        tag_map[i] = last_index + 1
        last_index += 1

    # shuffle the data
    percentage_split = 0.8
    data = list(zip(input_words, output_tags))
    np.random.shuffle(data)
    train_indices_len = int(np.round(percentage_split * len(data)))
    train_data = data[:train_indices_len]
    test_data = data[train_indices_len:]

    model = BiLSTM_CRF(len(word_map), tag_map, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    with torch.no_grad():
        print(model(data[0][0]))

    for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in data[:500]:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.

            targets = tags.view(tags.size(0))

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        print(model(data[0][0]))