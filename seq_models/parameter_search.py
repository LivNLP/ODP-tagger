# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 17/10/19 
# @Contact: michealabaho265@gmail.com

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append('/users/phd/micheala/Documents/Github/Health-outcome-tagger/')
import data_prep as dp
import helper_functions as utils
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
from seq_models.model import NNModel
from seq_models.biLSTM import BiLstm_model
import pickle
import itertools as it
import time

def load_data(source, hidden_dim, dropout_rate, embedding_dim):
    embeddings_file = open('../word_vecs/bionlp_wordvec/pubmed_1.pickle', 'rb')
    pubmed_embeddings = pickle.load(embeddings_file)

    word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag = dp.prepare_training_data(source)
    #pubmed_embeddings = utils.fetch_embeddings(file='../word_vecs/bionlp_wordvec/wikipedia-pubmed-and-PMC-w2v.bin', word_map=word_map, type='pubmed')

    model_configurations = {'embedding_dim': embedding_dim,
                            'hidden_size': hidden_dim,
                            'vocab_size': num_words,
                            'pos_size': num_pos,
                            'tag_map': tag_map,
                            'dropout_rate': dropout_rate,
                            'pos':True,
                            'weights_tensor': pubmed_embeddings,
                            'use_pretrained_embeddings': True,
                            'crf': False,
                            'neural': 'lstm'}

    batching_configurations = {'pairs':line_pairs,
                               'word_map':word_map,
                               'pos_map':pos_map,
                               'tag_map':tag_map}

    other_config = {'index2word':index2word,
                    'index2tag':index2tag,
                    }

    return model_configurations, batching_configurations, other_config

def batch_up_sets(batch_config, other_config, sampling=None, sampling_technique=None):
    percentage_split = 0.8
    input, target, max_length = dp.inputAndOutput(**batch_config)
    data = list(zip(input, target))
    #shuffle data
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
    if sampling:
        sampled_train_data, orig_labels_count, sampled_labels_count = dp.data_resample(list(inp), list(out), sampling, **other_config, sampling_technique=sampling_technique)
        sample_label_frequencies = [sampled_labels_count[i] for i in list(batch_config['tag_map'].keys())]
        initial = pd.DataFrame([[k, v] for k, v in orig_labels_count.items()], columns=['Label', 'Count'])
        older = pd.DataFrame([[k, v] for k, v in sampled_labels_count.items()], columns=['Label', 'Count'])
        print('Before and After sampling \n{}'.format(pd.merge(initial, older, on='Label')))
        sample_label_weights = [utils.class_balanced_loss(len(train_data), i, loss='chen_huang') for i in sample_label_frequencies]
        #utils.saveToPickle(train_data, val_data, label_weights, '../ebm-data/data_set_sampled.pickle')
        return sampled_train_data, val_data, sample_label_weights

def obj_function(data_source, sampling, sampling_technique='under', balanced_loss=False, loss_method=None, gamma=None,  tune=None):
    #phase 1
    if tune == 'phase1':
        lrs = config['learning-rate']
        bs = config['batch-size']
        epoch_set = config['epochs']
        parameters = list(it.product(lrs, bs, epoch_set))
    elif tune == 'phase2':
        parameters = config['sampling_percentage']
    elif tune == 'phase3':
        hidden_dim = config['hidden_dim']
        embeddin_dim = config['embedding_dim']
        dropout_rate = config['dropout_rate']
        parameters = list(it.product(hidden_dim, embeddin_dim, dropout_rate))
    start_outcome_token = 0
    loss_criterion = nn.CrossEntropyLoss()

    with open('parameter-performance-phase2.txt', 'w') as store_parameter:
        for p in parameters:
            if tune == 'phase1':
                lr, b_s, epochs = p
                us = [50]
            elif tune == 'phase2':
                lr, b_s, epochs = 0.1, 300, 60
                us = [p]
            elif tune == 'phase3':
                lr, b_s, epochs = 0.1, 300, 60
                us = [p]

            # load data
            model_configurations, batching_configurations, other_config = load_data(data_source)

            # load_model and beging training it on the loaded data
            print('\nTRAINING BEGINS\n')
            model = NNModel(model_configurations).to(dp.device)
            # initialize weights
            for param, values in model.named_parameters():
                if param.__contains__('weight'):
                    torch.nn.init.xavier_normal_(values)
                elif param.__contains__('bias'):
                    torch.nn.init.constant_(values, 0.0)

            # define backprop strategy
            optimizer = optim.SGD(model.parameters(), lr=lr)
            epoch_loss, epoch_accuracy, epoch_count = 0, 0, 1
            average_accuracy, average_loss = [], []
            print('lr: {} batch_size {} epochs {} us {}'.format(lr, b_s, epochs, us[0]))
            store_parameter.write('lr: {} batch_size: {} epochs: {} us: {}'.format(lr, b_s, epochs, us[0]))

            for u in us:
                # undersample raining data to create some balance
                data = open('../ebm-data/dataset.pickle', 'rb')
                data_loaded = pickle.load(data)
                train_data, val_data, label_weights = data_loaded['train_full'], data_loaded['val_data'], data_loaded['weights_full']
                #train_data, val_data, label_weights = batch_up_sets(batching_configurations, other_config, sampling=sampling, sampling_technique=sampling_technique)
                target_weights = torch.FloatTensor([np.round(i, 5) for i in label_weights])
                target_weights = target_weights.to(dp.device)
                loss_calculator = nn.CrossEntropyLoss(weight=target_weights) if balanced_loss else nn.CrossEntropyLoss()
                time_taken = 0
                model.train()
                for epoch in range(epochs):
                    batch_loss, batch_acc = 0, 0
                    batch_count = 0
                    start = time.time()
                    for batch_pairs in utils.batch_generator(train_data[:4000], b_s):
                        acc, loss = 0, 0
                        for batch in batch_pairs:
                            model.zero_grad()
                            hidden = model.hidden
                            sentence, pos, tags = batch[0][0].to(dp.device), batch[0][1].to(dp.device), batch[1].to(dp.device)
                            if 'crf' in model_configurations and model_configurations['crf'] == True:
                                l, feats = model(sentence, pos, hidden, tags)
                                path_score, decoded_best_path = model.decode(feats)
                                predicted = [i.item() for i in decoded_best_path]
                            else:
                                tag_scores = model(sentence, pos, hidden, tags)
                                if loss_method == 'focal' and gamma != None:
                                    # print('+++++++++++++++++++Using focal loss+++++++++++++++++++++++++++++++++=')
                                    l = utils.focal_loss(tag_scores, tags.view(tags.size(0)), target_weights, gamma)
                                else:
                                    l = loss_calculator(tag_scores, tags.view(tags.size(0)))
                                topv, topi = tag_scores.topk(1)
                                predicted = [i.item() for i in topi]

                            target = [i.item() for i in tags]
                            #accuracy
                            a = utils.token_level_accuracy(target, predicted)
                            acc += a
                            #loss
                            loss += np.round(l.item(), 4)
                            # print('accuracy {} Loss {}'.format(a, np.round(l.item(), 4)))

                            l.backward()
                            optimizer.step()

                        # print('Batch Average accuracy:- {:.3f} and Average loss:- {:.3f}'.format((acc / batch_size), (loss / batch_size)))
                        batch_acc += acc / b_s
                        batch_loss += loss / b_s
                        batch_count += 1

                    epoch_loss += batch_loss / batch_count
                    epoch_accuracy += batch_acc / batch_count
                    average_accuracy.append((batch_acc / batch_count))
                    average_loss.append((batch_loss / batch_count))
                    time_taken += (time.time() - start)
                    #print('Epoch {} acc {:.3f} and epoch loss {:.3f}. Duration {:.1f}'.format(epoch_count, (batch_acc / batch_count), (batch_loss / batch_count), (time.time() - start)))
                    epoch_count += 1
                model.eval()
                evaluation_accuracy, evaluation_loss, F_measure, AP = evaluate(model, val_data[:1000], model_configurations, other_config, loss_calculator, final_eval=False)

                print('\nEvaluation accuracy {}, Evaluation loss {}, F1 score {}, Average Precision {}\n'.format(evaluation_accuracy, evaluation_loss, F_measure, AP))
                store_parameter.write('Evaluation accuracy {}, Evaluation loss {}, F1 score {}, Average Precision {}\n'.format(evaluation_accuracy, evaluation_loss, F_measure, AP))
                epoch_count += 1

        return float(np.mean(average_accuracy)), float(np.mean(average_loss))


def evaluate(model, test_data, model_config, other_config, loss_criterion, final_eval=False):
    P,T,probs = [],[],[]
    with torch.no_grad():
        acc, batch_count, loss = 0, 0, 0
        actual_sentences, predicted_sentences = [], []
        for batch in test_data:
            sentence, pos, tags = batch[0][0].to(dp.device), batch[0][1].to(dp.device), batch[1].to(dp.device)
            sentence_len = len(tags)
            hidden = model.hidden

            if 'crf' in model_config and model_config['crf'] == True:
                l, feats = model(sentence, pos, hidden, tags)
                path_score, decoded_best_path = model.decode(feats)
                predicted = [i.item() for i in decoded_best_path]
            else:
                model_output = model(sentence, pos, hidden, tags)
                topv, topi = model_output.topk(1)
                predicted = [i.item() for i in topi]
                l = loss_criterion(model_output, tags.view(tags.size(0)))
                proba_pred = topv.to('cpu').numpy()
                for i in proba_pred[:, 0]:
                    probs.append(i)

            target = [i.item() for i in tags]
            target_sentence, predicted_sentence = utils.extract_predictions_from_model_output(target, predicted, other_config['index2tag'])
            actual_sentences.append(target_sentence)
            predicted_sentences.append(predicted_sentence)

            # accuracy
            acc += utils.token_level_accuracy(target, predicted)
            batch_count += 1
            # loss
            loss += np.round(l.item(), 4)

            P.append(predicted_sentence.split())
            T.append(target_sentence.split())

        evaluation_accuracy = float(acc/batch_count)
        evaluation_loss = float(loss/batch_count)

        P = [i for j in P for i in j]
        T = [i for j in T for i in j]
        classes = [v for k, v in other_config['index2tag'].items()]
        F_measure, AP, fig, classification = utils.metrics_score(target=T, predicted=P, cls=classes, prfs=True, probs=probs)
        if final_eval:
            return actual_sentences, predicted_sentences, evaluation_accuracy, F_measure, AP, classification, fig
    return evaluation_accuracy, evaluation_loss, F_measure, AP

if __name__ == '__main__':

    config = {'learning-rate':[0.1],
              'batch-size': [250],
              'epochs':[60],
              'sampling_percentage': [50],
              'hidden_dim': [128, 256, 512],
              'embedding_dim': [75, 150, 300],
              'dropout_rate': [0.1, 0.2, 0.5]}

    file_path = '../ebm-data/stanford'
    train_data_file = os.path.join(file_path, 'train_stanford_ebm.bmes')

    obj_function(data_source=train_data_file,
                 sampling=50,
                 sampling_technique='under',
                 balanced_loss=True,
                 loss_method=None,
                 gamma=None,
                 tune='phase2')


# def obj_function(lr, momentum, batch):
#     b = 250
#     # encd = model['encoder']
#     # decd = model['decoder']
#     use_attn = True
#     start_outcome_token = 0
#
#     embeddings_file = open('../word_vecs/bionlp_wordvec/pubmed_1.pickle', 'rb')
#     pubmed_embeddings = pickle.load(embeddings_file)
#
#     sampled_train_data, num_words, num_pos, tag_map = generate_data()
#
#     bilstm_pubmed = BiLstm_model(embedding_dim=150, hidden_size=256, vocab_size=num_words,
#                                  pos_size=num_pos,
#                                  output_size=len(tag_map), dropout_rate=0.2, weights_tensor=pubmed_embeddings,
#                                  use_pretrained_embeddings=True).to(dp.device)
#
#     # define backprop strategy
#     optimizer = optim.SGD(bilstm_pubmed.parameters(), lr=lr, momentum=momentum)
#     loss_criterion = nn.CrossEntropyLoss()
#
#     epoch_loss, epoch_accuracy, epoch_count = 1, 0, 0
#     average_accuracy, average_loss = [], []
#     for epoch in range(2):
#         epoch_batch_loss, epoch_batch_acc = 0, 0
#         batch_count = 0
#         for batch_pairs in utils.batch_generator(sampled_train_data[:200], b):
#             batch_acc, batch_loss = 0, 0
#             predicted_tags = []
#             errors, error_type = 0, []
#             for batch in batch_pairs:
#                 bilstm_pubmed.zero_grad()
#                 bilstm_pubmed.hidden = bilstm_pubmed.init_hidden()
#                 sentence, pos, tags = batch[0][0].to(dp.device), batch[0][1].to(dp.device), batch[1].to(dp.device)
#                 tag_scores = bilstm_pubmed(sentence, pos)
#
#                 print(tag_scores.size(), tags.size())
#                 l = loss_criterion(tag_scores, tags.view(tags.size(0)))
#
#                 topv, topi = tag_scores.topk(1)
#                 predicted = [i.item() for i in topi]
#                 target = [i.item() for i in tags]
#
#                 a = utils.token_level_accuracy(target, predicted)
#                 batch_acc += a
#
#                 batch_loss += np.round(l.item(), 4)
#                 # print('accuracy {} Loss {}'.format(a, np.round(l.item(), 4)))
#
#                 l.backward()
#                 optimizer.step()
#
#             #print('Batch Average accuracy:- {} and Average loss:- {}'.format((batch_acc / b), (batch_loss / b)))
#
#             batch_acc += batch_acc / b
#             batch_loss += batch_loss / b
#             batch_count += 1
#
#         epoch_loss += batch_loss / batch_count
#         epoch_accuracy += batch_acc / batch_count
#         average_accuracy.append((batch_acc / batch_count))
#         average_loss.append((batch_loss / batch_count))
#
#         print('{} epoch acc {} epoch loss {}'.format(epoch_count, (batch_acc / batch_count)*100, (batch_loss / batch_count)))
#
#         epoch_count += 1
#
#         return float(np.mean(average_accuracy)), float(np.mean(average_loss))
#
# def optimizer_func(obj_function, pbounds):
#     optimizer = BayesianOptimization(f=obj_function,
#                                      pbounds=pbounds,
#                                      verbose=2,
#                                      random_state=1)
#
#     optimizer.maximize(init_points=2, n_iter=3)
#
#     print(optimizer.max)