import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score, auc, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, precision_score, recall_score
from gensim.models import KeyedVectors
import data_prep as data_loader
from colored import fg, bg, attr
from pprint import pprint
import torch
import torch.nn.functional as F
from docx import Document
from docx.shared import RGBColor
import argparse
plt.style.use('ggplot')

def build_line_plots(save_path, values, title, labels=[], yticks=None, xticks=None):
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
    fig.savefig(os.path.join(save_path, title))
    return plt

def visualize_model(accuracy, average_loss, model_name, plt_dir, yticks=None, xticks=None):
    # visualize the training
    fig, (acc,lss) = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)
    if type(accuracy) == tuple:
        training_acc, eval_acc = accuracy
        x = range(0, len(training_acc))
        acc.plot(x, training_acc, color='green', label='train_accuracy')
        acc.plot(x, eval_acc, color='blue', label='eval_accuracy')
    else:
        x = range(0, len(accuracy))
        acc.plot(x, accuracy, color='green', label='train accuracy')
    plt.plot()
    acc.set_title('Token level Accuracy')
    acc.set_xlabel('Epochs')
    acc.set_ylabel('Accuracy')
    acc.legend()
    lss.plot(x, average_loss, color='red', label='training loss')
    lss.set_title('Average Loss')
    lss.set_xlabel('Epochs')
    lss.set_ylabel('Loss')
    lss.legend()
    fig.savefig(os.path.join(plt_dir, '{}'.format(model_name)))

def batch_generator(training_tensors, batch_size):
    training_tensors_size = len(training_tensors)
    for index in range(0, training_tensors_size, batch_size):
        yield training_tensors[index:min(index+batch_size, training_tensors_size)]


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
        store_embs = open(os.path.dirname(file) + '/{}.pickle'.format(type), 'wb')
        with open(file, 'r') as f:
            word_vecs = f.readlines()
        f.close()

    elif type.lower() == 'fasttext':
        store_embs = open(os.path.dirname(file) + '/{}.pickle'.format(type), 'wb')
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

def extract_predictions_from_model_output(target_tags, predicted_tags, index2tag):
    predicted = [index2tag[i] for i in predicted_tags]
    target = [index2tag[i] for i in target_tags]
    target_sentence = ' '.join(target)
    predicted_sentence = ' '.join(predicted)
    return target_sentence, predicted_sentence

def token_level_accuracy(target, predicted):
    sim, total = 0, 0
    for i,j in zip(target, predicted):
        if i == j:
            sim += 1
        total += 1
    return np.round(float(sim/total), 4)

def metrics_score(target, predicted, cls, probs=None, prfs=False):
    if set(target) == set(predicted):
        print('\n\n\nhere\n\n\n')
        print(set(cls))
        print(set(predicted))
        classification = classification_report(target, predicted, target_names=cls)
    else:
        classification = classification_report(target, predicted, labels=sorted(list(set(predicted))))
    if prfs:
        #precision, recall, f_scores, support = precision_recall_fscore_support(target, predicted)
        F_measure = f1_score(target, predicted, average='macro')
        recall = recall_score(target, predicted, average=None)
        precision = precision_score(target, predicted, average=None)
        #precision, recall, thresholds = precision_recall_curve(target, probs)
        #AUC = auc(recall, precision)

        AP = precision_score(target, predicted, average='macro')

        #fpr, tpr, thresholds = roc_curve(target, probs)

        fig, pr = plt.subplots()
        pr.plot(recall, precision, marker='+')
        pr.set_title('Precision-recall Curve')
        plt.legend()

        return F_measure, AP, fig, classification
    return classification

#creating a directory for the plots
def create_directories_per_series_des(name=''):
    _dir = os.path.abspath(os.path.join(os.path.curdir, name))
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir

#load data
def load_and_process(file_path, evaluation=False):

    word_map, word_count, index2word, num_words = data_loader.create_vocabularly(file_path)
    line_pairs, outputs = data_loader.readwordTag(file_path)
    tag_map = data_loader.create_tag_map(outputs)
    input_words, output_tags, max_len = data_loader.inputAndOutput(line_pairs, word_map, tag_map)
    if evaluation:
        return word_map, index2word, line_pairs, outputs, tag_map, input_words, output_tags, num_words, max_len
    else:
        return line_pairs, word_map, index2word, outputs, tag_map, input_words, output_tags, num_words, max_len

#class balanced loss
def class_balanced_loss(N, n, loss=None):
    if loss == 'yin':
        beta = (N-1)/N
        beta_n = beta ** n
        scaling_factor = (1-beta)/(1-beta_n)

    elif loss == 'chen_huang':
        beta = 1/np.sqrt(n)
        scaling_factor = beta

    elif loss == 'inverse':
        beta = 1/n
        scaling_factor = beta

    return float(scaling_factor)

#focal loss
def focal_loss(predicted, target, alpha, gamma, size_average=False):
    logpt = F.log_softmax(predicted, dim=1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.data.exp()

    at = torch.tensor([alpha[i].item() for i in target], dtype=torch.float32)
    at = at.to(device=data_loader.device)
    logpt = logpt * at

    loss = (-1 * (1-pt)**gamma) * logpt

    if size_average:
        return loss.mean()
    else:
        return loss.sum()

#save input and output to picleickle
def saveToPickle(train_data, val_data, label_weights, storage_location, sampled_train_data=None, sample_label_weights=None):
    in_out = {'train_full':train_data, 'train_sample':sampled_train_data if sampled_train_data else '', 'val_data':val_data,
              'weights_full':label_weights, 'weights_sample':sample_label_weights if sample_label_weights else ''}
    with open(storage_location, 'wb') as s:
        pickle.dump(in_out, s)
        s.close()

#change the 0 label to O
def change_0_O(file):
    file_name = os.path.basename(file).split('.')
    file_path = os.path.dirname(file)
    with open(file, 'r') as w, open(os.path.join(file_path, file_name[0]+'_1.txt'), 'w') as t:
        for i in w.readlines():
            if i == '\n':
                t.write('\n')
            else:
                v = i.split()
                if v[1].strip() == "0":
                    t.write('{} O\n'.format(v[0]))
                else:
                    t.write('{} {}\n'.format(v[0], v[1]))

def fetch_sentences(file):
    x = []
    with open(file, 'r') as f:
        y = []
        for i in f:
            if i == '\n':
                x.append([j for j in y])
                y.clear()
            else:
                y.append(i)
    return x

def visuzailize_output(actual_file, predicted_file):
    actual = fetch_sentences(actual_file)
    predicted = fetch_sentences(predicted_file)
    print(len(actual), len(predicted))
    s = o = 0
    X, Y = [], []
    for sent_1, sent_2 in zip(actual, predicted):
        sent__1 = [i.strip().split() for i in sent_1]
        sent__2 = [i.strip().split() for i in sent_2]
        b_pain = [i for i in sent__1 if i[1].strip() == 'O']
        o += len(b_pain)
        sent__2 = [i if len(i) == 2 else [i[0], i[-1]] for i in sent__2]
        if len(sent_1) != len(sent_2):
            s = 0
            Xx, Yy = [], []
            for i, j in enumerate(sent__1):
                if i in [i for i,j in enumerate(sent__2)]:
                    if j[0] == sent__2[i][0]:
                        #print(i, j, sent__2[i])
                        Xx.append(j)
                        Yy.append(sent__2[i])
                    else:
                        print('\t\t\t\t\t\t\t\t\t\t\t', j[0], '')
                        k = i
                        #se = sent__1[i+1:]
                        for n in range(i+1, len(sent__1)):
                            if sent__1[n][0] == sent__2[k][0]:
                                #print('\t\t\t\t',sent__1[n], sent_2[i])
                                s = n
                                Xx.append(sent__1[n])
                                Yy.append(sent__2[k])
                                k+=1
                            else:
                                #print('\t\t\t\t\t\t\t\t\t\t\t', sent__1[n], '')
                                pass
                        break
                else:
                    print(i, j[0], '')
            if Xx and Yy:
                X.append(Xx)
                Y.append(Yy)
        else:
            X.append(sent__1.copy())
            Y.append(sent__2.copy())
    print(X[:2])
    print(Y[:2])
    actual_final = [i[1] for j in X for i in j]
    predicted_final = [i[1] for j in Y for i in j]

    cls = sorted(list(set(actual_final)))
    print(list(set(actual_final)))
    print('absent', [i for i in actual_final if i not in predicted_final])
    F_measure, AP, fig, classification = metrics_score(actual_final, predicted_final, cls, prfs=True)
    print(F_measure, AP)
    print(classification)
    print(o)
    return actual, predicted

def error_position(d):
    start, mid_way, end = 0, 0, 0
    for pos, entity in enumerate(d):
        if entity[0] == 'bad-prediction':
            start += 1
        elif entity[-1] == 'bad-prediction':
            end += 1
        else:
            mid_way += 1
    start = float(start/len(d))
    mid_way = float(mid_way/len(d))
    end = float(end/len(d))
    return start, mid_way, end

def check_predictions(file_1, file_2, non_label, dest_folder, actual_with_predicted=False):
    actual = fetch_sentences(file_1)
    predicted = fetch_sentences(file_2)
    file_name = os.path.basename(file_1).split('.')

    if actual_with_predicted:
        #strip egdges
        actual = [[i.strip() for i in j] for j in actual]
        predicted = [[i.strip() for i in j] for j in predicted]
        predicted = [[i.split() for i in j] for j in predicted]
        predicted = [[i[0]+' '+i[-1] for i in j] for j in predicted]
    no_struggles = [0, [], []]
    struggles = [0, [], []]

    with open(os.path.join(dest_folder, 'analysis.txt'), 'w') as q:
        for i,j in zip(actual, predicted):
            pred = ''
            if i == j:
                print(' '.join([x.split()[0].strip() for x in i]))
                for y in j:
                    y_split = y.split()
                    if y_split[1].strip() != str(non_label):
                        f = '\033[1m'+'{} {}'.format(fg(4), y_split[0].strip())+'\033[0m'
                        c = '{}'.format(f)
                        pred += c
                        no_struggles[1].append(y_split[0])
                    else:
                        pred += '{} {}'.format(fg(0), y_split[0].strip())
                        no_struggles[2].append(no_struggles[1].copy())
                        no_struggles[1].clear()
                no_struggles[0] += 1
                print(pred)
                q.write('{}\n'.format(pred))
            else:
                if len(i) != len(j):
                    pass
                    # print(i)
                    # print(j)
                    # j_ = [i.split() for i in j]
                    # for count,x in enumerate(i):
                    #     x_split = x.split()
                    #     if x_split[0] == j_[count][0]:
                    #         print(x, j_[count])
                    #     else:
                    #         break
                else:
                    print(' '.join([x.split()[0].strip() for x in i]))
                    for x,y in zip(i,j):
                        x_split, y_split = x.split(), y.split()
                        if x_split[1].strip() != str(non_label):
                            if x_split[1] == y_split[1]:
                                f = '\033[1m' + '{} {}'.format(fg(4), y_split[0].strip()) + '\033[0m'
                                c = '{} {}'.format(x_split[1].strip(), f)
                                struggles[1].append(y_split[0])
                            else:
                                f = '\033[1m' + '{} {}'.format(fg(1), y_split[0].strip()) + '\033[0m'
                                c = '{} {}'.format(x_split[1].strip(), f)
                                struggles[1].append('bad-prediction')
                            pred += c
                        else:
                            pred += '{} {} {}'.format(x_split[1].strip(), fg(0), y_split[0].strip())
                            struggles[2].append(struggles[1].copy())
                            struggles[1].clear()
                    print(pred)
                    q.write('{}\n'.format(pred))
                struggles[0] += 1
        print('\n')
        print('Total number of sentences {}'.format(len(actual)))
        print('Total Number of sentences correctly predicted {}'.format(no_struggles[0]))
        well_predicted = [i for i in no_struggles[2] if i != []]
        count_well_predicted = [len(i) for i in well_predicted]
        print('Average entity length of correct predictions {}\nMax entity length {}\nMin entity length {}'.format(\
                np.mean(count_well_predicted), np.max(count_well_predicted), np.min(count_well_predicted)))

        print('\n')
        print('Total Number of sentences where predictor struggles {}'.format(struggles[0]))
        un_well_predicted = [i for i in struggles[2] if i != []]
        start, mid_way, end = error_position(un_well_predicted)
        print('Error at start {}, Error midway {}, Error at the end {}'.format(start, mid_way, end))
        count_un_well_predicted = [len(h) for h in un_well_predicted]
        un_well_predicted_good = [i for i in un_well_predicted if 'bad-prediction' not in i]
        un_well_predicted_bad = [i for i in un_well_predicted if 'bad-prediction' in i]
        print('un_well_predicted_good', len(un_well_predicted_good))
        #pprint(un_well_predicted)
        print('un_well_predicted_bad', len(un_well_predicted_bad))
        #pprint(un_well_predicted_bad)
        print('Average entity length of un_well_predicted_good {}\nMax entity length of un_well_predicted_good {}\nMin entity length of un_well_predicted_good {}\nAverage entity length un_well_predicted_bad {}\nMax entity length of un_well_predicted_bad {}\nMin entity length of un_well_predicted_bad {}'\
              ''.format( \
            np.mean([len(h) for h in un_well_predicted_good]),
            np.max([len(h) for h in un_well_predicted_good]),
            np.min([len(h) for h in un_well_predicted_good]),
            np.mean([len(h) for h in un_well_predicted_bad]),
            np.max([len(h) for h in un_well_predicted_bad]),
            np.min([len(h) for h in un_well_predicted_bad])))
        r = sent_length_analysis(well_predicted+un_well_predicted)
        r.savefig(os.path.join(dest_folder, file_name[0]+'.png'))
    #print(r)
def sent_length_analysis(sentences):
    hfont = {'fontname': 'Helvetica', 'fontweight': 'bold'}
    a = {}
    for i,j in enumerate(sentences):
        if len(j) not in a:
            s = []
            s.append(j)
            a[len(j)] = s
        elif len(j) in a:
            a[len(j)].append(j)
    plotting_a = {}
    print(plotting_a)
    for e in a:
        z = [i for i in a[e] if 'bad-prediction' not in i]
        plotting_a[e] = len(z)/len(a[e])
        # x.append(e)
        # y.append(len(z)/len(a[e]))
    plotting_a = dict(sorted(plotting_a.items(), key=lambda x:x[0]))
    plotting_b = {1: 0.8505, 2: 0.8571, 3: 0.7802, 4: 0.7189, 5: 0.4320, 6: 0.3681, 7: 0.2, 9: 0.0, 10: 0.0}

    print(plotting_a)
    # print(x)
    # sss = pd.DataFrame({'x':x, 'y':y}, columns=['x', 'y'])
    # ax = sns.lineplot(x="x", y="y", data=sss)
    plt.plot(list(plotting_a.keys()),list(plotting_a.values()), marker='*', color='green', label='Fine-tuning')
    plt.plot(list(plotting_b.keys()), list(plotting_b.values()), marker='+', color='magenta', label='Feature-extraction')
    plt.xticks(np.arange(1, 11, 1), **hfont)
    plt.yticks(**hfont)
    plt.xlabel('entity span length', **hfont)
    plt.ylabel('prediction accuracy', **hfont)
    plt.legend(prop={"weight":'bold'})

    return plt

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--file_1", default='comet-data/transformer_data/dev.txt', type=str, help="The input data dir. Should contain the training files.")
    parser.add_argument("--file_2", default='output/Feature_extraction/Comet_w2v_H_O_T/eval_preds.txt', type=str, help="pretrained embeddings file.")
    parser.add_argument("--dataset", default='Comet', required=True)
    parser.add_argument("--visualize_classification", action='store_true')
    args = parser.parse_args()

    k = 1
    if args.visualize_classification:
        visuzailize_output(args.file_1, args.file_2)
    else:
        non_label = 'O' if args.dataset.lower() == 'comet' else 0
        dest_folder = os.path.dirname(args.file_2)
        check_predictions(file_1=args.file_1,
                          file_2=args.file_2,
                          non_label=non_label,
                          dest_folder=dest_folder,
                          actual_with_predicted=True)
        # check_predictions(file_1='ebm-data/transformer-data/full/test.txt',
        #                   file_2='output/Feature_extraction/EBM_bert_H_O_T/decoded.out',
        #                   non_label=0,
        #                   dest_folder='output/Fine-tuning/EBM_BIOBERT/',
        #                   actual_with_predicted=True)
        # check_predictions(file_1='ebm-data/transformer-data/full/test.txt',
        #                   file_2='output/Fine-tuning/EBM_BIOBERT/test_predictions.txt',
        #                   non_label=0,
        #                   dest_folder='output/Fine-tuning/EBM_BIOBERT/',
        #                   actual_with_predicted=True)

        # check_predictions(file_1='comet-data/transformer_data/dev.txt',
        #                   file_2='output/Fine-tuning/COMET_BIOBERT/eval_preds.txt',
        #                   non_label='O',
        #                   dest_folder='output/Fine-tuning/COMET_BIOBERT/',
        #                   actual_with_predicted=True)
