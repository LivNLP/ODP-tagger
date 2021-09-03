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
from pathlib import Path, PurePath
import argparse
import json
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spacyModel():
    current_file_location = Path(os.path.realpath(__file__))
    spacy_model_loc = os.path.join(current_file_location.parents[1], 'pico-outcome-prediction/Trained_Tagger')

    #load required nlp models
    spacy_model = spacy.load(spacy_model_loc)
    spacy_model.add_pipe(spacy_model.create_pipe('sentencizer'))
    return spacy_model

start_outcome_token, end_outcome_token = 0, 1

def create_vocabularly(data_input, data_output):
    #define a tokenizer borrowing keras processing
    word_map, pos_map = {}, {}
    word_count, pos_count = {}, {}
    index2word, index2pos = {0: "SOO_token", 1:"EOO_token"}, {0: "SOP_token", 1:"EOP_token"}

    num_words, num_pos = 2, 2
    #creating a vocabularly that merges the training and validation data otherwise (else block) only the training data
    if type(data_input) in [tuple, list]:
        all_sentences = []
        for i in data_input:
            with open(i, 'r') as f:
                f_lines = f.readlines()
                for j in f_lines:
                    all_sentences.append(re.split('\s+', j.strip()))
                f.close()
    else:
        with open(data_input, 'r') as g:
            g_lines = g.readlines()
            all_sentences = [re.split('\s+', i.strip()) for i in g_lines]
            g.close()

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

    with open(os.path.join(data_output, 'vocab.json'), 'w') as vocab, open(os.path.join(data_output, 'pos.json'), 'w') as pos_:
        json.dump(word_map, vocab, indent=2)
        json.dump(pos_map, pos_, indent=2)
        vocab.close()

    return word_count, index2word, num_words, pos_count, index2pos, num_pos

def reverse_dict(x):
    return dict((v,k) for k,v in x.items())

def readwordTag(data_input, mode=''):
    for item in data_input:
        if item.__contains__(mode):
            f = open(item, 'r')
            lines = f.readlines()
            f.close()
    line_pairs, seq, tags = [], [], []
    for line in lines:
        line = re.split('\s+', line.strip())
        #only word and tag
        if len(line) == 2:
            seq.append(line[0])
            tags.append(line[1])
        #taking wod and pos as features and the tag
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

def prepare_tensor_pairs(file_path, file_out, mode=''):
    line_pairs, outputs = readwordTag(file_path, mode=mode)
    size = len(line_pairs)
    if mode.strip().lower() == 'train':
        word_count, index2word, num_words, pos_count, index2pos, num_pos = create_vocabularly(file_path, file_out)
        word_map = reverse_dict(index2word)
        pos_map = reverse_dict(index2pos)
        tag_map = create_tag_map(outputs)
        index2tag = reverse_dict(tag_map)
        with open(os.path.join(file_out, 'tags.json'), 'w') as t:
            json.dump(tag_map, t, indent=2)
    else:
        with open(os.path.join(file_out, 'vocab.json') , 'r') as v, open(os.path.join(file_out, 'pos.json'), 'r') as p, open(os.path.join(file_out, 'tags.json'), 'r') as t:
            word_map = json.load(v)
            pos_map = json.load(p)
            tag_map = json.load(t)
        return line_pairs, word_map, tag_map, pos_map, size
    return word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag, size


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

def inputAndOutput(pairs, word_map, pos_map, tag_map, encoded_outputs=False, test=False):
    input_batch = [x[0] for x in pairs]
    output_batch = [x[1] for x in pairs]
    long_sequences = []
    if all(type(feat) == tuple for item in input_batch for feat in item):
        for i in input_batch:
            if len(i) > 128:
                v = [x for x,y in i]
                long_sequences.append(' '.join(v))
        input_batch = [[(word_map[i], pos_map[j]) for i,j in ins] for ins in input_batch]
    else:
        input_batch = [[word_map[i] for i in ins] for ins in input_batch]

    if encoded_outputs:
        output_batch = [[encoded_outputs[i] for i in j] for j in output_batch]
    else:
        output_batch = [[tag_map[i] for i in j] for j in output_batch]

    max_seq_length = max([len(i) for i in input_batch])
    input_features = prepareTensor(input_batch)
    output_features = [prepare_tensor(i) for i in output_batch]
    return input_features, output_features, max_seq_length, long_sequences

def oov_vocab(input_batch, word_map):
    words_in_test = list(set([i[0] if len(i) > 1 else i for j in input_batch for i in j]))
    trained_vocabularly = list(word_map.keys())
    oov_words = [i for i in words_in_test if i not in trained_vocabularly]
    return oov_words

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

    for i in new_seq_data:
        if all(type(n) == tuple for n in i):
            tokens, pos = [j[0] for j in i], [j[1] for j in i]
            final_seq_data.append((prepare_tensor(tokens), prepare_tensor(pos)))
        else:
            tokens = i
            final_seq_data.append(prepare_tensor(tokens))
    padded_sentences = final_seq_data
    # padded = nn.utils.rnn.pad_sequence(seq_datatrain_stanford_ebm.bmes)
    # padded_array = padded.cpu().numpy()
    # padded_sentences = [list(i) for i in list(zip(*padded_array))]
    return padded_sentences

def prepare_tensor(x):
    return torch.tensor(x, dtype=torch.long, device=device).view(-1, 1)

def split_data(file):
    #
     #splitting the lengthy abstracts in the files into sentences, using the built spacy model
    #
    file_name = os.path.basename(file).split('.')
    file_name = file_name[0] + '_ebm.bmes'

    #file_name = os.path.abspath(os.path.join('BIO-data', file_name))
    spacy_model = spacyModel()
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
            doc = spacyModel(u.strip())
            instances = []
            for i, sent in enumerate(doc.sents):
                tokens = str(sent).split()
                char_case_features = [i[0].isupper() for i in tokens]
                char_case_features = ['T' if i == True else 'F' for i in char_case_features]
                pos_features = [i.pos_ for i in spacyModel(str(sent))]
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


def reduce_features(file, dest, ebm=False, default_line=False, test=False):
    file_name = os.path.basename(file).split('.')[0]
    sentences = []
    with open(file, 'r') as f:
        sentence = []
        for line in f.readlines():
            if line == '\n':
                sentences.append([i for i in sentence])
                sentence.clear()
            else:
                if default_line == True:
                    sentence.append(line)
                else:
                    line_split = re.split('\s+',line)
                    sentence.append(str(line_split[0])+' '+str(line_split[1]))
        f.close()

    dest = utils.create_directories_per_series_des(dest)

    if ebm:
        print(len(sentences))
        print(max([len(i) for i in sentences]))
        np.random.shuffle(sentences)
        if test:
            write_to(os.path.join(dest, 'test.txt'), sentences)
        else:
            p = int(np.round(0.5*len(sentences)))
            train = sentences[:p]
            test = sentences[p:]

            write_to(os.path.join(dest, 'dev.txt'), train)
            write_to(os.path.join(dest, 'test.txt'), test)
    else:
        write_to(os.path.join(dest, '{}.txt'.format(file_name)), sentences)


def write_to(file, source):
    with open(file, 'w') as tr:
        if type(source) == list:
            print('yes')
            for sent in source:
                for pair in sent:
                    tr.write('{}\n'.format(pair))
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


def bc2gm(folder):
    files = glob('{}/*.json'.format(folder))
    file_dest = Path(files[0])
    file_dest = file_dest.parent
    spacy_model = spacyModel()
    print(files)
    for f in files:
        t = os.path.basename(f).split('.')[0]
        with open(f, 'r') as b, open(os.path.join(file_dest, '{}.bmes'.format(t)), 'w') as c:
            data = json.load(b)
            e = 0
            for i in data:
                text = data[i]["sent"]
                gene_tag = data[i]["gene"]
                gene_tag = [list(range(i[0], i[-1]+1)) for i in gene_tag]

                pos = []
                for i in text:
                    p = [i.pos_ for i in spacy_model(i)]
                    pos.append(p[0])

                x = 0
                for m,n in enumerate(text):
                    if m == x:
                        if m in [i for u in gene_tag for i in u]:
                            tags = gene_tag[0]
                            for o,h in enumerate(tags):
                                if h == m and o == 0:
                                    c.write('{} B-gene {}\n'.format(text[h], pos[h]))
                                else:
                                    c.write('{} I-gene {}\n'.format(text[h], pos[h]))
                            x = tags[-1]
                            gene_tag = gene_tag[1:]
                        else:
                            c.write('{} 0 {}\n'.format(n, pos[m]))
                        x += 1
                c.write('\n')
                e += 1
        print('{} has {} instances'.format(f, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data source')
    parser.add_argument('--outputdir', type=str, required=True)
    parser.add_argument('--ebm', action='store_true')
    parser.add_argument('--default_line', action='store_true')
    parser.add_argument("--function", default=None, type=str, required=True, help="Source of training, validation and test data")
    args = parser.parse_args()
    
    # fetch embeddings
    if args.function.lower() in ['fetch_embeddings', 'fetch embeddings']:
        dir = os.path.abspath(args.data)
        file_path = [os.path.join(dir, i) for i in os.listdir(dir) if i.lower().__contains__('train') or i.lower().__contains__('dev')]
        dest = utils.create_directories_per_series_des(args.outputdir)
        word_count, index2word, num_words, pos_count, index2pos, num_pos = create_vocabularly(file_path, dest)
        word_map = reverse_dict(index2word)
        weights = utils.fetch_embeddings(file='word_vecs/bionlp_wordvec/wikipedia-pubmed-and-PMC-w2v.bin', dest=dest, word_map=word_map, type='pubmed')
    # fetch outcokme lexicon
    elif args.function.lower() in ['fetch_outcome_lexicon', 'fetch outcome lexicon']:
        fetch_outcome_lexicon('ebm-data/Review Outcomes - Mental Health .xlsx', args.data)

    elif args.function.lower() in ['reduce features', 'reduce_features']:
        reduce_features(file=args.data, dest=args.outputdir, ebm=args.ebm, default_line=args.default_line)

    # fetch bc2gm dataset
    elif args.function.lower in ['bc2gm']:
        print(args)
        bc2gm(args.data)
