# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 24/08/19
# @Contact: michealabaho265@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report



def build_line_plots(values, title, labels=[], yticks=None, xticks=None):
    fig, ax = plt.subplots()
    if xticks != None:
        xloc = ticker.MultipleLocator(base=xticks)
        ax.xaxis.set_major_locator(xloc)
    if yticks != None:
        yloc = ticker.MultipleLocator(base=yticks)
        ax.yaxis.set_major_locator(yloc)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    plt.plot(values)
    return plt

def batch_generator(training_tensors, batch_size):
    training_tensors_size = len(training_tensors)
    for index in range(0, training_tensors_size, batch_size):
        yield training_tensors[index:min(index+batch_size, training_tensors_size)]


def fetch_embeddings(file, word_map, embedding_dim):
    weights = {}
    oov_words = 0
    with open(file, 'r') as f:
        for i in f.readlines():
            line = i.split(" ")
            word = line[0]
            if word not in weights:
                try:
                    vector = np.asarray(line[1:], dtype='float32')
                except Exception as e:
                    print(e)
                weights[word] = vector[:embedding_dim]
        f.close()

    word_map_list = list(word_map.items())
    word_map_list.insert(0, ('SOO_token', 1))
    word_map_list.insert(1, ('EOO_token', 1))
    target_word_map = dict(word_map_list)
    weights_tensor = np.zeros((len(target_word_map), embedding_dim))

    for word,id in target_word_map.items():
        if word in weights:
            vec= weights.get(word)
            weights_tensor[id] = vec
        else:
            vec = np.random.rand(embedding_dim)
            weights_tensor[id] = vec
            oov_words += 1
            print('OOV_word {}'.format(word))
            print(vec)

    return torch.FloatTensor(weights_tensor)

def extract_predictions_from_model_output(model_output, true_tags, index2tag):
    topv, topi = model_output.data.topk(1)
    predicted_sequence = [i.item() for i in topi]
    predicted_sentence_list = [index2tag[i] for i in predicted_sequence]
    target = [index2tag[i] for i in true_tags.tolist()]
    return target, predicted_sentence_list

def token_level_accuracy(target, predicted):
    sim, total = 0, 0
    for i,j in zip(target, predicted):
        if i == j:
            sim += 1
        total += 1
    return sim, total

def metrics_score(target, predicted, cls, prfs=False, classification=False):
    if prfs:
        p,r,f,s = precision_recall_fscore_support(target, predicted)
        return p,r,f,s
    return classification_report(target, predicted, target_names=cls)



