# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 10/12/19 
# @Contact: michealabaho265@gmail.com
import pickle
import numpy as np
import data_prep as dp
import helper_functions as utils
import pandas as pd

def load_data(source, args):
    embeddings_file = open(args.pretrained_embeddings, 'rb')
    pubmed_embeddings = pickle.load(embeddings_file)

    if type(source) in [tuple, list]:
        word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag, train_size = dp.prepare_training_data(source)
    else:
        word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag = dp.prepare_training_data(source)
    #pubmed_embeddings = utils.fetch_embeddings(file='../word_vecs/bionlp_wordvec/wikipedia-pubmed-and-PMC-w2v.bin', word_map=word_map, type='pubmed')

    model_configurations = {'embedding_dim': args.embedding_dim,
                            'hidden_size': args.hidden_dim,
                            'vocab_size': num_words,
                            'pos_size': num_pos,
                            'tag_map': tag_map,
                            'dropout_rate': args.dropout_rate,
                            'pos':args.pos,
                            'weights_tensor': pubmed_embeddings,
                            'use_pretrained_embeddings': args.use_pretrained_embeddings,
                            'crf': args.crf_layer,
                            'neural':args.neural}

    batching_configurations = {'pairs':line_pairs,
                               'word_map':word_map,
                               'pos_map':pos_map,
                               'tag_map':tag_map}

    other_config = {'index2word':index2word,
                    'index2tag':index2tag,
                    'train_size':train_size if type(source) in [tuple, list] else None}

    return model_configurations, batching_configurations, other_config

def batch_up_sets(batch_config, other_config, args, percentage_split=0.8, test=False):
    input, target, max_length = dp.inputAndOutput(**batch_config)
    data = list(zip(input, target))

    #testing dataset
    if test == True:
        return data
    #shuffle data
    if 'train_size' not in other_config:
        np.random.shuffle(data)
    train_indices_len = int(np.round(percentage_split * len(data)))
    train_data = data[:train_indices_len]
    val_data = data[train_indices_len:]

    #training data
    inp, out = zip(*train_data)
    labels_count = dp.count_labels(data=list(out), ix_map=other_config['index2tag'])
    label_frequencies = [labels_count[i] for i in list(batch_config['tag_map'].keys())]
    label_weights = [utils.class_balanced_loss(len(train_data), i, loss='chen_huang') for i in label_frequencies]
    #utils.saveToPickle(train_data, val_data, label_weights, '../ebm-data/data_set_full.pickle')
    # sampling the training data
    if args.sampling:
        sampled_train_data, orig_labels_count, sampled_labels_count = dp.data_resample(list(inp), list(out), args.sampling, other_config['index2word'], other_config['index2tag'], sampling_technique=args.sampling_technique)
        sample_label_frequencies = [sampled_labels_count[i] for i in list(batch_config['tag_map'].keys())]
        initial = pd.DataFrame([[k, v] for k, v in orig_labels_count.items()], columns=['Label', 'Count'])
        older = pd.DataFrame([[k, v] for k, v in sampled_labels_count.items()], columns=['Label', 'Count'])
        print('Before and After sampling \n{}'.format(pd.merge(initial, older, on='Label')))
        sample_label_weights = [utils.class_balanced_loss(len(train_data), i, loss='chen_huang') for i in sample_label_frequencies]
        #utils.saveToPickle(train_data, val_data, label_weights, '../ebm-data/data_set_sampled.pickle')
        return sampled_train_data, val_data, sample_label_weights

    return train_data, val_data, label_weights

