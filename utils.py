import pickle
import torch
import numpy as np
import data_prep as dp
import helper_functions as utils
import pandas as pd
import re
import json
from glob import glob
from pathlib import Path
import os
import argparse
from pprint import pprint
from pathlib import Path, PurePath

def load_data(args, mode):
    if args.use_pretrained_embeddings and not args.use_bert_features:
        embeddings_file = open(os.path.join(args.pretrained_embeddings,'pubmed.pickle'), 'rb')
        pubmed_embeddings = pickle.load(embeddings_file)

    elif args.use_pretrained_embeddings and args.use_bert_features:
        #pubmed_embeddings = None
        files = [j for j in glob('{}/*.pickle'.format(args.pretrained_embeddings))]
        pubmed_embeddings, x, size = {}, 1, 0
        for i in files:
            if os.path.basename(i).__contains__(mode):
                print('\t Loading {} embeddings data'.format(os.path.basename(i).split('.')[0]))
                read_file = pickle.load(open(i, 'rb'))
                pubmed_embeddings = read_file
    else:
        pubmed_embeddings = None

    data_files = os.listdir(args.data)
    data_files = [j for j in glob('{}/*.{}'.format(args.data, args.extension))]
    if mode.lower() == 'train':
        word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag, size = dp.prepare_tensor_pairs(file_path=data_files,
                                                                                                                                      file_out=args.outputdir,
                                                                                                                                      mode=mode)
    else:
        df_files = [j for j in data_files if j.__contains__(mode.lower())]
        line_pairs, word_map, tag_map, pos_map, size = dp.prepare_tensor_pairs(file_path=df_files,
                                                                               file_out=args.outputdir,
                                                                               mode=mode)
        batching_configurations = {'pairs': line_pairs,'word_map': word_map,'pos_map': pos_map,'tag_map': tag_map}
        return batching_configurations

    model_configurations = {'embedding_dim': args.embedding_dim,
                            'hidden_size': args.hidden_dim,
                            'vocab_size': num_words,
                            'pos_size': num_pos,
                            'tag_map': tag_map,
                            'word_map': word_map,
                            'dropout_rate': args.dropout_rate,
                            'pos': args.pos,
                            'weights_tensor': pubmed_embeddings,
                            'use_pretrained_embeddings': args.use_pretrained_embeddings,
                            'use_bert_features': args.use_bert_features,
                            'crf': args.crf_layer,
                            'neural': args.neural}

    batching_configurations = {'pairs': line_pairs,
                               'word_map': word_map,
                               'pos_map': pos_map,
                               'tag_map': tag_map}

    other_config = {'index2word': index2word,
                    'index2tag': index2tag,
                    'train_size': size if type(args.data) in [tuple, list] else None}

    return model_configurations, batching_configurations, other_config

def batch_up_sets(args, batch_config, other_config=None, mode=''):
    input_features, output_features, max_seq_length, long_sequences = dp.inputAndOutput(**batch_config)
    mode_data = list(zip(input_features, output_features))

    if mode.lower() != 'train':
        print('\n{} data-set Length {}\n'.format(mode.upper(), len(mode_data)))
        return mode_data

    print('\n{} data-set Length {}\n'.format(mode.upper(), (len(mode_data))))

    #shuffle data
    #np.random.shuffle(mode_data)

    #if validation dataset not passed, portion off some data for validation
    if args.train_val_split != None:
        train_indices_len = int(np.round(args.train_val_split * len(mode_data)))
        train_data = mode_data[:train_indices_len]
        val_data = mode_data[train_indices_len:]
        print('Training data Length:', len(train_data), 'Validation data Length:', len(val_data))

    print('MAX-sequence Length', max_seq_length)

    #obtaining class specific weights to pass to a cost sensitive function
    inp, out = zip(*mode_data)

    # sampling the training data
    if args.sampling_percentage:
        sampled_train_data, orig_labels_count, sampled_labels_count = dp.data_resample(list(inp), list(out), args.sampling_percentage, other_config['index2word'], other_config['index2tag'], sampling_technique=args.sampling_technique)
        sample_label_frequencies = [sampled_labels_count[i] for i in list(batch_config['tag_map'].keys())]
        initial = pd.DataFrame([[k, v] for k, v in orig_labels_count.items()], columns=['Label', 'Count'])
        older = pd.DataFrame([[k, v] for k, v in sampled_labels_count.items()], columns=['Label', 'Count'])
        print('Before and After sampling \n{}'.format(pd.merge(initial, older, on='Label')))
        sample_label_weights = [utils.class_balanced_loss(len(mode_data), i, loss='chen_huang') for i in sample_label_frequencies]
        #utils.saveToPickle(train_data, val_data, label_weights, '../ebm-data/data_set_sampled.pickle')
        return sampled_train_data, sample_label_weights
    else:
        labels_count = dp.count_labels(data=list(out), ix_map=other_config['index2tag'])
        label_frequencies = [labels_count[i] for i in list(batch_config['tag_map'].keys())]
        label_weights = [utils.class_balanced_loss(len(mode_data), i, loss='chen_huang') for i in label_frequencies]

    return mode_data, label_weights

# aggregate features extracted from bert
def bert_feature_aggregated(_path_, type='dev'):
    files_location = Path(_path_)
    pickle_file_output, count = {}, 1
    for file in files_location.iterdir():
        if file.is_file() and os.path.basename(file).__contains__(type):
            with open(os.path.join(_path_, file), 'r') as file_read, open('ebm-data/transformer-data/bert-output/{}.pickle'.format(type), 'wb') as byte_write:
                file_items = json.load(file_read)
                for i in file_items:
                    instance = file_items[i]
                    instance_keys = list(instance.keys())
                    f = []
                    d = 0
                    for c,j in enumerate(instance_keys):
                        if c == d:
                            vec = np.array(instance[j])
                            word = j.split('_*_')[0]
                            #check ahead for masked words
                            for k in range(c+1, len(instance_keys)):
                                if instance_keys[k].startswith('##'):
                                    word += instance_keys[k][2:].split('_*_')[0]
                                    vec = np.add(vec, np.array(instance[instance_keys[k]]))
                                    d = k
                                else:
                                    d += 1
                                    break
                            f.append((word, vec.tolist()))
                    pickle_file_output[count] = f
                    count += 1
                pickle.dump(pickle_file_output, byte_write)
                file_read.close()
                byte_write.close()


def fetch_bert_features(source1):
    #embs = torch.zeros((len(input_seq), 1, self.embedding_dim), device=dp.device)
    with open(source1, 'rb') as pickle_load1:
        features = pickle.load(pickle_load1)
        out_features = {}
        for i in features:
            for j in features[i]:
                print(j[0])
                print(type(j[1]))
                break
            break

def under_sample_extraction(args):
    model_configurations_train, batching_configurations_train, other_config_train = load_data(args, mode='train')

    # applying both sampling and reweighting classes using a class imbalanced loss
    train_data, val_data, label_weights = batch_up_sets(args=args,
                                                        batch_config=batching_configurations_train,
                                                        other_config=other_config_train,
                                                        mode='train')


    for i in [('train', train_data)]:
        with open('ebm-data/transformer-data/sampled/{}.txt'.format(i[0]), 'w') as file:
            for example in i[1]:
                example_text = example[0][0]
                example_tags = example[1]
                example_text = [i.item() for i in example_text]
                example_tags = [i.item() for i in example_tags]
                example_text = [other_config_train['index2word'][i] for i in example_text]
                example_tags = [other_config_train['index2tag'][i] for i in example_tags]
                for x, y in zip(example_text, example_tags):
                    file.write('{} {}\n'.format(x, y))
                file.write('\n')

if __name__ == '__main__':
    # load data
    data_dir = 'ebm-data/stanford/'
    output_dir = 'output/HOT'
    # set commandline parameters
    par = argparse.ArgumentParser()
    par.add_argument("--data", default=data_dir, type=str, help="Source of training, validation and test data")
    par.add_argument("--outputdir", default=output_dir, type=str, help="Source of training, validation and test data")
    par.add_argument("--embedding_dim", default=150, type=int, help="size of word embedding")
    par.add_argument("--hidden_dim", default=256, type=int, help="number of neurons in hidden layer")
    par.add_argument("--batch_size", default=250, type=int, help="batch_size")
    par.add_argument("--learning_rate", default=0.1, type=float, help="learning rate")
    par.add_argument("--n_epochs", default=100, type=int, help="epochs")
    par.add_argument("--dropout_rate", default=0.2, type=float, help="epochs")
    par.add_argument("--pos", action='store_true', help="'True'-use parts of speech or 'False'-ignore them")
    par.add_argument("--train_dev_test",default='train dev test', help="Turn on or off training, evaluation and prediction phases")
    par.add_argument("--pretrained_embeddings", default='../word_vecs/bionlp_wordvec/pubmed_1.pickle', type=str, help="source of pretrained embeddings")
    par.add_argument("--use_pretrained_embeddings", action='store_true', help="use or ignore pretrained embeddings")
    par.add_argument("--use_bert_features", action='store_true', help="use or ignore pretrained embeddings")
    par.add_argument("--crf_layer", action='store_true', help="use or ignore crf_layer")
    par.add_argument("--neural", default='lstm', type=str, help="LSTM or GRU or RNN")
    par.add_argument("--train_val_split", default=None, help="split the training data into train and validation sets")
    par.add_argument("--sampling_percentage", default=None, type=int, help="percentage to use when sampling down the majority calss labels")
    par.add_argument("--sampling_technique", default='under', type=str, help="sampling strategy")

    args = par.parse_args()
    args.outputdir = utils.create_directories_per_series_des(args.outputdir)
    #bert_feature_aggregated('/media/micheala/Mykel/output', type='test')
    #fetch_bert_features('ebm-data/transformer-data/bert-features/test.pickle')
    modes = args.train_dev_test.split()

    if 'train' in modes:
        model_configurations, batching_configurations_train, other_config_train = load_data(args, mode=modes[0])
        train_data, label_weights = batch_up_sets(args=args,
                                                 batch_config=batching_configurations_train,
                                                 other_config=other_config_train,
                                                 mode=modes[0])
    if 'dev' in modes:
        batching_configurations_eval = load_data(args, mode=modes[1])
        eval_data = batch_up_sets(args=args,
                                 batch_config=batching_configurations_eval,
                                 mode=modes[1])


    if 'test' in modes:
        batching_configurations_test = load_data(args, mode=modes[2])
        test_data = batch_up_sets(args=args,
                                  batch_config=batching_configurations_test,
                                  mode=modes[2])

