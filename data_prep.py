import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import numpy as np
import sys
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import re
import spacy
import os
import pandas as pd
import helper_functions as utils
from pprint import pprint
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_file_location = Path(__file__)

#load required nlp models
spacy_model = spacy.load(os.path.join(current_file_location.parents[1], 'pico-outcome-prediction/trained_tagger'))
spacy_model.add_pipe(spacy_model.create_pipe('sentencizer'))

start_outcome_token, end_outcome_token = 0, 1

def create_vocabularly(data_input):
    #define a tokenizer borrowing keras processing
    word_map, pos_map = {}, {}
    word_count, pos_count = {}, {}
    index2word, index2pos = {0: "SOO_token", 1:"EOO_token"}, {0: "SOP_token", 1:"EOP_token"}

    num_words, num_pos = 2, 2

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
            all_sentences = [re.split('\s+', i.strip()) for i in g_lines]
            if any(len(i) > 2 for i in all_sentences):
                all_sentences_words = [(s[0], s[2]) for s in all_sentences if len(s) >= 3]
            else:
                all_sentences_words = [s[0] for s in all_sentences if len(s) == 2]

    for feature in all_sentences_words:
        if type(feature) == tuple:
            word, pos = feature
            if word not in word_map:
                word_map[word] = num_words
                word_count[word] = 1
                index2word[num_words] = word
                num_words += 1
            else:
                word_count[word] += 1

            if pos not in pos_map:
                pos_map[pos] = num_pos
                pos_count[pos] = 1
                index2pos[num_pos] = pos
                num_pos += 1
            else:
                pos_count[pos] += 1
        else:
            word = feature
            if feature not in word_map:
                word_map[word] = num_words
                word_count[word] = 1
                index2word[num_words] = word
                num_words += 1
            else:
                word_count[word] += 1

    return word_count, index2word, num_words, pos_count, index2pos, num_pos

def reverse_dict(x):
    return dict((v,k) for k,v in x.items())


def readwordTag(data_input):
    if type(data_input) == str:
        f = open(data_input, 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = data_input

    line_pairs, seq, tags = [], [], []
    for line in lines:
        line = re.split('\s+', line.strip())
        if len(line) == 2:
            seq.append(line[0])
            tags.append(line[1])

        elif len(line) > 2:
            seq.append((line[0], line[2]))
            tags.append((line[1]))

        elif len(line) == 1:
            if seq and tags:
                s_seq = [i for i in seq]
                t_tags = [i for i in tags]
                line_pairs.append((s_seq, t_tags))
            seq.clear()
            tags.clear()

    tag_list = list(set([i for k,l in line_pairs for i in l]))

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

def inputAndOutput(pairs, word_map, pos_map, tag_map, encoded_outputs=False, test=False):
    input_batch = [x[0] for x in pairs]
    output_batch = [x[1] for x in pairs]
    oov_words = []
    if test:
        words_in_test = list(set([i for j in input_batch for i in j]))
        trained_vocabularly = list(word_map.keys())
        oov_words = [i for i in words_in_test if i not in trained_vocabularly]
        #largest_index = sorted(word_map.items(), key=lambda x:x[1])[-1][1]
        # for i, oov in enumerate(oov_words):
        #     i = i + 1
        #     word_map[oov] = largest_index + i
        #input_batch = [[word_map[i] if i in word_map  else 0 for i in j] for j in input_batch]

    if all(type(ins[0]) == tuple for ins in input_batch):
        input_batch = [[(word_map[i], pos_map[j]) for i,j in ins] for ins in input_batch]
    else:
        input_batch = [[word_map[i] for i in ins] for ins in input_batch]

    if encoded_outputs:
        output_batch = [[encoded_outputs[i] for i in j] for j in output_batch]
    else:
        output_batch = [[tag_map[i] for i in j] for j in output_batch]

    input_features, lit_len = prepareTensor(input_batch)
    output_features = [prepare_tensor(i) for i in output_batch]

    if oov_words:
        return input_features, output_features, lit_len, oov_words
    return input_features, output_features, lit_len

def prepareTensor(seq_data, attention=False):
    #append the end of outcome token at the end of every sequence
    new_seq_data, final_seq_data = [], []

    if attention == True:
        for i in seq_data:
            e = [j for j in i]
            e.append((end_outcome_token, end_outcome_token))
            new_seq_data.append(e)
    else:
        new_seq_data = seq_data

    longest_seq_data = 0
    for i in new_seq_data:
        if all(type(n) == tuple for n in i):
            tokens = [j[0] for j in i]
            pos = [j[1] for j in i]
            final_seq_data.append((prepare_tensor(tokens), prepare_tensor(pos)))
        else:
            final_seq_data.append(prepare_tensor(i))
        if longest_seq_data < len(i):
            longest_seq_data = len(i)

    padded_sentences = final_seq_data
    # padded = nn.utils.rnn.pad_sequence(seq_datatrain_stanford_ebm.bmes)
    # padded_array = padded.cpu().numpy()
    # padded_sentences = [list(i) for i in list(zip(*padded_array))]
    return padded_sentences, longest_seq_data

def prepare_tensor(x):
    return torch.tensor(x, dtype=torch.long, device=device).view(-1, 1)

def split_data(file):
    #
     #splitting the lengthy abstracts in the files into sentences, using the built spacy model
    #
    file_name = os.path.basename(file).split('.')
    file_name = file_name[0] + '_ebm.bmes'

    #file_name = os.path.abspath(os.path.join('BIO-data', file_name))

    num_instances = 0
    num_abstracts = 0

    with open(file, 'r') as f:
        file_input = f.readlines()
        with open(file_name, 'w') as sent_tokenize:
            tags, words = [], []
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
                            #
                            # in addition to a word, include two other features, part of speech and the whether or not the first character of the word is upper case or not
                            #
                            pos = [i.pos_ for i in spacy_model(str(sent))]
                            s = str(sent).split()
                            char_case_feature = [i[0].isupper() for i in s]
                            char_case_feature = ['T' if i == True else 'F' for i in char_case_feature]
                            t = tags[:len(s)]
                            if s and t:
                                for x, y, z, o in zip(s, t, pos, char_case_feature):
                                    sent_tokenize.writelines(x + ' ' + y + ' ' + z + ' ' + o)
                                    sent_tokenize.writelines('\n')
                                tags = tags[len(s):]
                            num_instances += 1
                            sent_tokenize.writelines('\n')
                        num_abstracts += 1
                    tags.clear()
                    words.clear()
            sent_tokenize.close()
        f.close()

    return file_name, num_abstracts, num_instances

def fetch_outcome_lexicon(mental_path, dataset):
    outcomes = {}
    df = pd.read_excel(mental_path)

    outcomes['Mental'] = df['outcomeDomainLabel'].tolist()

    dir_name = os.path.dirname(dataset)
    base_name = os.path.basename(dataset)
    print(dir_name, base_name)
    new_name = base_name.split('.')

    new_file = new_name[0]+'_1.'+new_name[1]

    with open(os.path.join(dir_name, new_file), 'w') as t, open(dataset, 'r') as f:
        for i in f.readlines():
            if i == '\n':
                t.writelines('\n')
            else:
                v = i.strip().split()
                if v[0].lower():
                    if v[0].lower() in [str(i).lower() for i in outcomes['Mental']]:
                        v.append(1)
                    else:
                        v.append(-1)
                else:
                    if v[0].lower() in [str(i).lower() for i in outcomes['Mental']]:
                        v.append(1)
                    else:
                        v.append(-1)

                t.writelines(' '.join([str(i) for i in v]))
            t.writelines('\n')

    return outcomes

def prepare_data(train_file, test_file, train_percentage, test_percentage):
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
            with open(trainfile, 'r') as f, open(testfile, 'r') as g:
                file1_input, file2_input = f.readlines(), g.readlines()
                folder_name = os.path.dirname(trainfile)

                s = open('Train_stanford_ebm.bmes', 'w')
                t = open('Dev_stanford_ebm.bmes', 'w')
                u = open('Test_stanford_ebm.bmes', 'w')
                v = open('Raw_stanford_ebm.bmes', 'w')
                q = open('stanford_config.txt', 'w')

                train = create_dict_of_training_instances(file1_input)

                test = create_dict_of_training_instances(file2_input)

                trainset_size = np.round(train_percentage * len(train.keys()))
                write_to_training_files(train, s, t, trainset_size)

                testset_size = np.round((1 - test_percentage) * len(test.keys()))
                write_to_training_files(test, u, v, testset_size)

                q.writelines('Total number of abstracts in Original train data:- {}\n'.format(num_train_abstracts))
                q.writelines('Total number of instraces in Original train data:- {}\n'.format(num_train_instances))
                q.writelines('Total number of instraces in final working train data:- {}\n'.format(trainset_size))
                q.writelines('Total number of instraces in final working validation data:- {}\n'.format(
                    len(train.keys()) - trainset_size))

                q.writelines('Total number of abstracts in Original test data:- {}\n'.format(num_test_abstracts))
                q.writelines('Total number of instraces in Original test data:- {}\n'.format(num_test_instances))
                q.writelines('Total number of instraces in final working test data:- {}\n'.format(testset_size))
                q.writelines('Total number of instraces in final working raw/inference data:- {}'.format(
                    len(test.keys()) - testset_size))

                f.close()
                g.close()
                q.close()


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

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def data_resample(input, output, percentage, index2word, index2tag, sampling_technique):
    #count current output

    labels_count = count_labels(data=output, ix_map=index2tag)
    print('The current count per label \n {}'.format(labels_count))
    if sampling_technique.lower() == 'smote':
        t = [i for j in range(len(input)) for i in input[j].tolist()]
        r = [i for j in range(len(output)) for i in output[j].tolist()]

        t = np.array(t)
        r = np.array(r).reshape(len(r))

        sm = SMOTE(random_state=42)
        x, y = sm.fit_resample(t, r)

        input_tokens = [index2word[i] for i in x.reshape(len(x)).tolist()]
        tags = [index2tag[i] for i in y.reshape(len(y)).tolist()]
        print(Counter(tags))

        with open('../ebm-data/sample_smote/re_sampled_smote.txt', 'w') as resample:
            for i in input_tokens:
                resample.writelines(str(i).strip())
                resample.writelines('\n')
            resample.close()

        all_text = ' '.join(input_tokens).split()
        se = open('../ebm-data/sample_smote/re_sentenced_smote.txt', 'w')
        line_pairs = []
        average_sent_length = []

        #setting maximum sentence length to 50
        for u in chunks(all_text, n=50):
            u = ' '.join(u)
            doc = spacy_model(u.strip())
            instances = []
            for i, sent in enumerate(doc.sents):
                tokens = str(sent).split()
                char_case_features = [i[0].isupper() for i in tokens]
                char_case_features = ['T' if i == True else 'F' for i in char_case_features]
                pos_features = [i.pos_ for i in spacy_model(str(sent))]
                labels = tags[:len(tokens)]
                average_sent_length.append(len(tokens))

                for token_feature, label, pos_feature, char_feature in zip(tokens, labels, pos_features, char_case_features):
                    instances.append((token_feature, char_feature, pos_feature, label))
                    se.writelines(token_feature+' '+label+' '+pos_feature+' '+char_feature)
                    se.writelines('\n')
                se.writelines('\n')
                tags = tags[len(tokens):]
            if instances:
                line_pairs.append(instances)
        se.close()

    elif sampling_technique.lower() == 'under':
        target_data_u = []
        out_tensor_u = []
        total_size = 0
        for i,j in zip(input, output):
            if (all(k.item() == 0 for k in j)):
                target_data_u.append((i, j))
            else:
                out_tensor_u.append((i, j))
            total_size += 1

        print('Number of instances in training set before under sampling: {}'.format(total_size))
        print('Number of instances with only 0 label or No label tag {}'.format(len(target_data_u)))

        indices = np.random.permutation(int(np.round(float(percentage/100) * len(target_data_u))))
        target_data_u = [target_data_u[i] for i in indices]

        line_pairs_u = out_tensor_u + target_data_u

        print('Number of instances in training set after under sampling {}'.format(len(line_pairs_u)))
        sampled_labels = [i[1] for i in line_pairs_u]
        sampled_labels_count = count_labels(data=sampled_labels, ix_map=index2tag)
        print('The current count per label \n{}'.format(sampled_labels_count))
        np.random.shuffle(line_pairs_u)

        return line_pairs_u, labels_count, sampled_labels_count
    return se.name, line_pairs

def count_labels(data, ix_map):
    labels_ = [i.item() for j in data for i in j]
    labels_count = Counter([ix_map[o] for o in labels_])
    return labels_count

def prepare_training_data(file_path):
    if type(file_path) == tuple or type(file_path) == list:
        file_path = [i for i in file_path if i.lower().__contains__('train') or i.lower().__contains__('dev')]
        file_path = sorted(file_path, key=lambda x:len(x), reverse=True)
        train_pairs, o = readwordTag(file_path[0])
        file_path = merge_train_val(file_path)
    if file_path:
        word_count, index2word, num_words, pos_count, index2pos, num_pos = create_vocabularly(file_path)
        line_pairs, outputs = readwordTag(file_path)
        word_map = reverse_dict(index2word)
        pos_map = reverse_dict(index2pos)
        tag_map = create_tag_map(outputs)
        index2tag = reverse_dict(tag_map)
    if type(file_path) == tuple or type(file_path) == list:
        train_size = len(train_pairs)
        return word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag, train_size
    return word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag

def merge_train_val(files_list):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_source.bmes')
    with open(file_path, 'w') as n:
        for j in files_list:
            d = open(j, 'r')
            for r in d.readlines():
                n.write(r)
        n.close()
    return file_path

def batch_processor(input_batch):
    input = input_batch[0]
    target_tensor = input_batch[1].to(device)
    if type(input) == tuple:
        input_tensor = input[0].to(device)
        pos_tensor = input[1].to(device)
        return input_tensor, pos_tensor, target_tensor, input_tensor.size(0), target_tensor.size(0)
    else:
        input_tensor = input.to(device)
        return input_tensor, target_tensor, input_tensor.size(0), target_tensor.size(0)


def reduce_features(file, dest, test=False):
    file_path = os.path.dirname(os.path.abspath(file))
    file_name = os.path.basename(file).split('_')[0]
    sentences = []
    with open(file, 'r') as f:
        sentence = []
        for line in f.readlines():
            if line == '\n':
                sentences.append([i for i in sentence])
                sentence.clear()
            else:
                line_split = line.split(' ')
                sentence.append(str(line_split[0])+' '+str(line_split[1]))
        f.close()

    dest = utils.create_directories_per_series_des(dest)
    if test:
        write_to(os.path.join(dest, 'test.txt'), sentences)
    else:
        p = int(np.round(0.8*len(sentences)))
        train = sentences[:p]
        test = sentences[p:]

        write_to(os.path.join(dest, 'train.txt'), train)
        write_to(os.path.join(dest, 'dev.txt'), test)


def write_to(file, source):
    with open(file, 'w') as tr:
        if type(source) == list:
            for sent in source:
                for pair in sent:
                    tr.write(pair)
                    tr.write('\n')
                tr.write('\n')
        elif type(source) == str:
            op = open(source, 'r')
            for sent in op.readlines():
                if sent == '\n':
                    tr.write('\n')
                else:
                    line = sent.split(' ')
                    tr.write(line[0]+' '+line[1])
                    tr.write('\n')
            op.close()
        tr.close()



if __name__ == '__main__':
    #fetch embeddings
    # file_path = '../pico-outcome-prediction/corrected_outcomes/BIO-data/stanford'
    # train_data_file = os.path.join(file_path, 'train_stanford_ebm.bmes')
    # # test_data_file = os.path.join(file_path, 'Test_stanford_ebm.bmes')
    # word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag = prepare_training_data(train_data_file)
    # #fetch_outcome_lexicon('ebm-data/Review Outcomes - Mental Health .xlsx', train_data_file)
    # weights = utils.fetch_embeddings(file='word_vecs/bionlp_wordvec/wikipedia-pubmed-and-PMC-w2v.bin', word_map=word_map, type='pubmed')

    #create dataset
    file_path = 'ebm-data/stanford'
    dest_path = '../transformers/glue_data/ebm-data'
    train_data_file = os.path.join(file_path, 'test_stanford_ebm_org.bmes')

    reduce_features(train_data_file, dest_path, test=True)
    #write_to(os.path.join(file_path, 'test.txt'), test_data_file)
