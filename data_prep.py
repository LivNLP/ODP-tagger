import torch
import numpy as np
import torch.nn as nn
from keras_preprocessing import text
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
import os
import helper_functions as utils
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_outcome_token, end_outcome_token = 0, 1

def create_vocabularly(data_input):
    #define a tokenizer borrowing keras processing
    word_map = {}
    word_count = {}
    index2word = {0: "SOO", 1:"EOO"}
    num_words = 2
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
    input_batch = [x[0] for x in pairs]
    output_batch = [x[1] for x in pairs]
    oov_words = []
    if test:
        words_in_test = list(set([i for j in input_batch for i in j]))
        oov_words = [i for i in words_in_test if i not in list(word_map.keys())]
        largest_index = sorted(word_map.items(), key=lambda x:x[1])[-1][1]
        # for i, oov in enumerate(oov_words):
        #     i = i + 1
        #     word_map[oov] = largest_index + i
        input_batch = [[word_map[i] if i in word_map  else 0 for i in j] for j in input_batch]
        los_length = max([len(s) for s in input_batch])
    else:
        input_batch = [[word_map[i] for i in j] for j in input_batch]
        los_length = max([len(s) for s in input_batch])
    if encoded_outputs:
        output_batch = [[encoded_outputs[i] for i in j] for j in output_batch]
    else:
        output_batch = [[tag_map[i] for i in j] for j in output_batch]

    input_tensors, lit_len = prepareTensor(input_batch)
    output_tensors, lot_len = prepareTensor(output_batch)

    if oov_words:
        return input_tensors, output_tensors, lit_len, oov_words
    else:
        return input_tensors, output_tensors, lit_len

def prepareTensor(seq_data):
    #append the end of outcome token at the end of every sequence
    new_seq_data = []
    for i in seq_data:
        e = [j for j in i]
        e.append(end_outcome_token)
        new_seq_data.append(e)

    seq_data = [torch.tensor(i, dtype=torch.long, device=device).view(-1, 1) for i in new_seq_data]
    longest_seq_data = max([t.size(0) for t in seq_data])
    padded_sentences = seq_data
    # padded = nn.utils.rnn.pad_sequence(seq_datatrain_stanford_ebm.bmes)
    # padded_array = padded.cpu().numpy()
    # padded_sentences = [list(i) for i in list(zip(*padded_array))]
    return padded_sentences, longest_seq_data

def split_data(file):
    #
     #splitting the lengthy abstracts in the files into sentences, using the built spacy model
    #
    spacy_model = spacy.load(os.path.abspath('../trained_tagger'))
    spacy_model.add_pipe(spacy_model.create_pipe('sentencizer'))
    file_name = os.path.basename(file)
    num_instances = 0
    num_abstracts = 0
    with open(file, 'r') as f:
        file_input  = f.readlines()
        sent_tokenize = open(file_name, 'w')
        tags,words = [],[]
        for i in range(len(file_input)):
            if file_input[i] != '\n':
                l = file_input[i].strip().split()
                words.append(l[0])
                tags.append(l[1])
            else:
                abstract = ' '.join(words)
                if abstract:
                    doc = spacy_model(abstract)
                    for i, sent in enumerate(doc.sents):
                        s = str(sent).split()
                        t = tags[:len(s)]
                        if s and t:
                            for x,y in zip(s, t):
                                sent_tokenize.writelines(x+' '+y)
                                sent_tokenize.writelines('\n')
                            tags = tags[len(s):]
                        num_instances += 1
                        sent_tokenize.writelines('\n')
                    num_abstracts += 1
                tags.clear()
                words.clear()
        f.close()

    return file_name, num_abstracts, num_instances

def prepare_data(train_file, test_file):
    #start by splitting the abstracts into sentences in both ntrain and test sets
    trainfile, num_train_abstracts, num_train_instances = split_data(train_file)
    testfile, num_test_abstracts, num_test_instances = split_data(test_file)
    print('Total number of abstracts in train data {}'.format(num_train_abstracts))
    print('Total number of instraces in train data {}'.format(num_train_instances))
    print('Total number of abstracts in test data {}'.format(num_test_abstracts))
    print('Total number of instraces in test data {}'.format(num_test_instances))

    train = {}
    test = {}

    current_dir_files = os.listdir('.')
    if trainfile in current_dir_files:
        if testfile in current_dir_files:

            with open('train_stanford.bmes', 'r') as f, open('test_stanford.bmes', 'r') as g:
                file1_input, file2_input = f.readlines(), g.readlines()

                s = open('train_stanford_ebm.bmes', 'w')
                t = open('dev_stanford_ebm.bmes', 'w')
                u = open('test_stanford_ebm.bmes', 'w')
                v = open('raw_stanford_ebm.bmes', 'w')

                train = create_dict_of_training_instances(file1_input)

                test = create_dict_of_training_instances(file2_input)

                train_percentage = np.round(0.8*len(train.keys()))
                write_to_training_files(train, s, t, train_percentage)

                test_percentage = np.round(0.8 * len(test.keys()))
                write_to_training_files(test, u, v, test_percentage)

                f.close()
                g.close()


def create_dict_of_training_instances(list_items):
    y, t, k = '', 0, {}
    for i in list_items:
        if i != '\n':
            y += i
            y += ' '
        else:
            if y:
                k[t] = y.strip()
                t += 1
            y = ''
    return k

def write_to_training_files(train_dict, file1, file2, split_percentage):
    for m, n in train_dict.items():
        if m < split_percentage:
            n = n.split('\n')
            for l in n:
                file1.writelines(l.strip())
                file1.writelines('\n')
            file1.writelines('\n')
        else:
            n = n.split('\n')
            for o in n:
                file2.writelines(o.strip())
                file2.writelines('\n')
            file2.writelines('\n')
    file1.close()
    file2.close()

if __name__ == '__main__':
    file_path = 'ebm-data/train_ebm.bmes'
    word_map, word_count, index2word = create_vocabularly(file_path)

    line_pairs, outputs = readwordTag(file_path)
    tag_map = create_tag_map(outputs)

    inp, out, max_sent_len = inputAndOutput(line_pairs, word_map, tag_map)

    print("Vocabularly size is {}".format(len(word_map)))
    print(utils.fetch_embeddings('/users/phd/micheala/Documents/Github/pico-back-up/glove.840B.300d.txt', word_map, 10))
    # word_map['']
    # print("Vocabularly {}".format(word_map))
    # print("Total number of tagged sentence instances {}".format(len(line_pairs)))
    # print('Longest sentence is {}'.format(max_sent_len))
    # word_count_sorted = list(sorted(word_count.items(), key=lambda x: x[1]))
    # print("Top most popular words {}".format(dict(word_count_sorted[-10:])))
    # print("Least most popular words {}".format(dict(word_count_sorted[:5])))
    # print("List of classes {}".format(tag_map))


