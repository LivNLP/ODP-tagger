# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 11/12/19
# @Contact: michealabaho265@gmail.com

import os
import argparse
import re
import numpy as np
import torch.nn as nn
import torch
import json
import pprint
import pickle
import sys
sys.path.append('../')
print(sys.path)
from gensim.models import KeyedVectors
import data_prep as dp
from glob import glob
from flair.embeddings import BertEmbeddings, FlairEmbeddings, FastTextEmbeddings, ELMoEmbeddings, StackedEmbeddings, ELMoTransformerEmbeddings, PooledFlairEmbeddings

from flair.data import Sentence
import helper_functions as utils


def fetch_embeddings(file, dest, word_map, type=None):
    weights = {}
    oov_words = 0

    if type.lower() == 'pubmed':
        store_embs = open(dest + '/{}.pickle'.format(type), 'wb')
        pubmed_vecs = KeyedVectors.load_word2vec_format(file, binary=True)
        model_vocab = list(pubmed_vecs.vocab.keys())
        word_vecs = []
        for i in model_vocab:
            d = i+' '+' '.join(str(i) for i in pubmed_vecs[i])
            word_vecs.append(d)

    elif type.lower() == 'glove':
        store_embs = open(dest + '/{}.pickle'.format(type), 'wb')
        with open(file, 'r') as f:
            word_vecs = f.readlines()
        f.close()

    elif type.lower() == 'fasttext':
        store_embs = open(dest + '/{}.pickle'.format(type), 'wb')
        with open(file, encoding='utf-8') as f:
            word_vecs = f.readlines()

    for i in word_vecs:
        line = i.split(" ")
        word = line[0]
        if word not in weights:
            try:
                vector = np.asarray(line[1:], dtype='float32')
            except Exception as e:
                print(e)
            weights[word] = vector

    weights_tensor = np.zeros((len(word_map), vector.shape[0]))

    for word,id in word_map.items():
        if word in weights:
            vec= weights.get(word)
            weights_tensor[id] = vec
        else:
            vec = np.random.rand(vector.shape[0])
            weights_tensor[id] = vec
            oov_words += 1

    word_vecs_tensors = torch.FloatTensor(weights_tensor)

    pickle.dump(word_vecs_tensors, store_embs)
    store_embs.close()

    return word_vecs_tensors

def use_flair_to_extract_context_embeddings(file, dest_folder, embedding_type, embedding_size, pretrained_model=None):
    if embedding_type.lower() == 'elmo':
        context_embedding = ELMoEmbeddings(model='pubmed')
    elif embedding_type.lower() == 'elmo_transformer':
        context_embedding = ELMoTransformerEmbeddings()
    elif embedding_type.lower() == 'flair':
        context_embedding = PooledFlairEmbeddings()
    elif embedding_type.lower() == 'bioflair':
        flair_1 = PooledFlairEmbeddings('pubmed-forward')
        flair_2 = PooledFlairEmbeddings('pubmed-backward')
        elmo = ELMoEmbeddings(model='pubmed')
        #bert = BertEmbeddings(bert_model_or_path='bert-base-multilingual-cased', layers='-1')
        context_embedding = StackedEmbeddings(embeddings=[flair_1, flair_2, elmo])
    elif embedding_type.lower() == 'biobert' or embedding_type.lower() == 'bert':
        context_embedding = BertEmbeddings(bert_model_or_path=pretrained_model, layers='-1')

    data = {}
    dest_name = os.path.basename(file).split('.')

    print(dest_folder)
    with open(file, 'r') as f, open('{}/{}.pickle'.format(dest_folder, dest_name[0]), 'wb') as d:
        sentence = ''
        instance = []
        j = 0
        for i in f.readlines():
            if i != '\n':
                i = i.split()
                sentence += ' '+i[0]
            elif i == '\n':
                sent = Sentence(sentence.strip())
                context_embedding.embed(sent)
                v = ''
                for i in sent:
                    instance.append((i.text, i.embedding[:embedding_size]))
                sentence = ''

                if instance:
                    data[j] = list(zip(*(instance.copy())))
                    j += 1
                instance.clear()
        pickle.dump(data, d)
        f.close()
        d.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,
                        help="The input data dir. Should contain the training files.")
    parser.add_argument("--file_name", default='word_vecs/bionlp_wordvec/wikipedia-pubmed-and-PMC-w2v.bin', type=str, required=True,
                        help="pretrained embeddings file.")
    parser.add_argument("--pretrained_model", default='bert-base-multilingual-cased')
    parser.add_argument("--mode", default='train', type=str, help="feture dimension")
    parser.add_argument("--embedding_type", default=None, required=True, type=str, help='which embeddings to create')
    parser.add_argument("--embedding_size", default=768, type=int, help='size of the embedding')
    parser.add_argument("--dest_folder", default=None, required=True, help='where to store the embeddings created')

    args = parser.parse_args()
    dest_folder = utils.create_directories_per_series_des(args.dest_folder)
    with open(os.path.join(dest_folder, 'config.txt'), 'w') as cf:
        for k, v in vars(args).items():
            cf.write('{}: {}\n'.format(k, v))
        cf.close()

    if args.embedding_type.lower() in ['pubmed', "glove", "fasttext"]:
        data_files = [j for j in glob('{}/*.bmes'.format(args.data_dir))]
        word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag, size = dp.prepare_tensor_pairs(
                                                                                                                file_path=data_files,
                                                                                                                file_out=dest_folder,
                                                                                                                mode=args.mode)
        weights = fetch_embeddings(file=args.file_name, dest=args.dest_folder, word_map=word_map, type=args.embedding_type)
    elif args.embedding_type.lower() in ["bert", "bioflair", "elmo", "biobert"]:
        use_flair_to_extract_context_embeddings(file=args.file_name,
                                                dest_folder=dest_folder,
                                                embedding_type=args.embedding_type,
                                                embedding_size=args.embedding_size,
                                                pretrained_model=args.pretrained_model)