import torch
import numpy as np
import torch.nn as nn
from keras_preprocessing import text
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_vocabularly(data_input):
    #define a tokenizer borrowing keras processing
    word_map = {}
    word_count = {}
    index2word = {}
    num_words = 0
    if type(data_input) == tuple:
        all_sentences = []
        for i in data_input:
            with open(i, 'r') as f:
                f_lines = f.readlines()
                all_sentences.append(f_lines)
                f.close()

        all_sentences = [line for list in all_sentences for line in list]
        all_sentences_words = [re.split('\s+', i.strip())[0] for i in all_sentences]
    else:
        with open(data_input, 'r') as g:
            g_lines = g.readlines()
            all_sentences_words = [re.split('\s+', i.strip())[0] for i in g_lines]
            g.close()

    for word in all_sentences_words:
        if word not in word_map:
            word_map[word] = num_words
            word_count[word] = 1
            index2word[num_words] = word
            num_words += 1
        else:
            word_count[word] += 1
    return word_map, word_count, index2word


def readwordTag(data_input):
    with open(data_input, 'r') as f:
        lines = f.readlines()
        line_pairs, seq, tags = [], [], []
        for line in lines:
            line = re.split('\s+', line.strip())
            if len(line) > 1:
                seq.append(line[0])
                tags.append((line[1]))

            elif len(line) == 1:
                if seq and tags:
                    s_seq = [i for i in seq]
                    t_tags = [i for i in tags]
                    line_pairs.append((s_seq, t_tags))
                seq.clear()
                tags.clear()

        tag_list = list(set([i for k,l in line_pairs for i in l]))
        f.close()
    return line_pairs, tag_list

def encod_outputs(outputs):
    encoded_outputs = {}
    o = OneHotEncoder()
    outputs = np.array(outputs).reshape(-1, 1)
    o.fit(outputs)
    outputs = o.transform(outputs).toarray()
    for output in outputs:
        d = o.inverse_transform([output])
        d = d.item(0)
        if d not in encoded_outputs:
            encoded_outputs[d] = output
    return encoded_outputs

def create_tag_map(outputs):
    tag_map = {}
    num_words = 0
    outputs = sorted(outputs)
    for o in outputs:
        if o not in tag_map:
            tag_map[o] = num_words
            num_words += 1
        else:
            pass
    return tag_map

def inputAndOutput(pairs, word_map, tag_map, encoded_outputs=False, test=False):
    input_batch = [x for x,y in pairs]
    output_batch = [y for x,y in pairs]
    oov_words = []
    if test:
        words_in_test = list(set([i for j in input_batch for i in j]))
        oov_words = [i for i in words_in_test if i not in list(word_map.keys())]
        largest_index = sorted(word_map.items(), key=lambda x:x[1])[-1][1]
        # for i, oov in enumerate(oov_words):
        #     i = i + 1
        #     word_map[oov] = largest_index + i
        input_batch = [[word_map[i] if i in word_map  else 0 for i in j] for j in input_batch]
        lengths = [len(i) for i in input_batch]
    else:
        input_batch = [[word_map[i] for i in j] for j in input_batch]
        lengths = max([len(i) for i in input_batch])
    if encoded_outputs:
        output_batch = [[encoded_outputs[i] for i in j] for j in output_batch]
    else:
        output_batch = [[tag_map[i] for i in j] for j in output_batch]

    if oov_words:
        return prepareTensor(input_batch), prepareTensor(output_batch), lengths, oov_words
    else:
        return prepareTensor(input_batch), prepareTensor(output_batch), lengths



def prepare_tensor(seq):
    return

def prepareTensor(seq_data):
    seq_data = [torch.tensor(i, dtype=torch.long, device=device).view(-1, 1) for i in seq_data]
    padded_sentences = seq_data
    # padded = nn.utils.rnn.pad_sequence(seq_data)
    # padded_array = padded.cpu().numpy()
    # padded_sentences = [list(i) for i in list(zip(*padded_array))]
    return padded_sentences


if __name__ == '__main__':
    file_path = 'ebm-data/train_ebm.bmes'
    word_map, word_count, index2word = create_vocabularly(file_path)

    line_pairs, outputs = readwordTag(file_path)
    tag_map = create_tag_map(outputs)

    inp, out, max_sent_len = inputAndOutput(line_pairs, word_map, tag_map)
    
    print("Vocabularly size is {}".format(len(word_map)))
    print("Total number of tagged sentence instances {}".format(len(line_pairs)))
    print('Longest sentence is {}'.format(max_sent_len))
    word_count_sorted = list(sorted(word_count.items(), key=lambda x: x[1]))
    print("Top most popular words {}".format(dict(word_count_sorted[-10:])))
    print("Least most popular words {}".format(dict(word_count_sorted[:5])))

    print("List of classes {}".format(tag_map))
    print(inp[1])
    print(inp[1].shape)
    print(tag_map)
