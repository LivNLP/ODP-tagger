# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 16/09/19
# @Contact: michealabaho265@gmail.com

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pandas as pd
import argparse
sys.path.append(os.path.abspath('../'))
import data_prep as dp
from seq_models.biLSTM import BiLstm_model
from seq_models.crf import crf
from seq_models.model import NNModel
import numpy as np
import helper_functions as utils
import utils as ut
import pickle
import time
#from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt

def train(model, train_data, model_config, other_config, pltdir, args, label_weights, val_data, balanced_loss=False, loss_method=None, gamma=None):

    target_weights = torch.FloatTensor([np.round(i, 5) for i in label_weights])
    target_weights = target_weights.to(dp.device)
    calculate_loss = nn.CrossEntropyLoss(weight=target_weights) if balanced_loss else nn.CrossEntropyLoss()

    #required parameters
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    average_accuracy, average_loss = [], []
    epoch_loss, epoch_accuracy, epoch_count  = 0, 0, 1
    best_accuracy, best_model, F1_scores = 0, [], []

    time_taken = 0
    for epoch in range(args.n_epochs):
        model.train()
        batch_acc,batch_loss,batch_count = 0,0,0
        start = time.time()
        for batches in utils.batch_generator(train_data, args.batch_size):
            acc, loss = 0,0
            # sente = [x for x,y in batches]
            # sentences = [i[0] for i in sente]
            # sent_pos = [i[1] for i in sente]
            # tags = [y for x,y in batches]
            # model.zero_grad()
            # hidden = model.hidden
            # tag_scores = model(sentences, sent_pos, hidden)
            #
            for batch in batches:
                model.zero_grad()
                hidden = model.hidden
                sentence, pos, tags = batch[0][0].to(dp.device), batch[0][1].to(dp.device), batch[1].to(dp.device)
                sd = [i.item() for i in sentence]
                #print(' '.join([other_config['index2word'][i] for i in sd]))
                if 'crf' in model_config and model_config['crf'] == True:
                    l, feats = model(sentence, pos, hidden, tags)
                    path_score, decoded_best_path = model.decode(feats)
                    predicted = [i.item() for i in decoded_best_path]
                else:
                    tag_scores = model(sentence, pos, hidden, tags)
                    if loss_method == 'focal' and gamma != None:
                        #print('+++++++++++++++++++Using focal loss+++++++++++++++++++++++++++++++++=')
                        l = utils.focal_loss(tag_scores, tags.view(tags.size(0)), target_weights, gamma)
                    else:
                        l = calculate_loss(tag_scores, tags.view(tags.size(0)))
                    topv, topi = tag_scores.topk(1)
                    predicted = [i.item() for i in topi]

                target = [i.item() for i in tags]
                a = utils.token_level_accuracy(target, predicted)
                acc += a

                loss += np.round(l.item(), 4)
                l.backward()
                optimizer.step()

            print('Batch Average accuracy:- {:.3f} and Average loss:- {:.3f}'.format((acc / args.batch_size), (loss / args.batch_size)))
            batch_acc += acc/args.batch_size
            batch_loss += loss/args.batch_size
            batch_count += 1

        epoch_loss += batch_loss/batch_count
        epoch_accuracy += batch_acc/batch_count
        average_accuracy.append((batch_acc/batch_count))
        average_loss.append((batch_loss/batch_count))
        time_taken += (time.time() - start)
        print('Epoch {} acc {:.3f} and epoch loss {:.3f}. Duration {:.1f}'.format(epoch_count, (batch_acc/batch_count)*100, (batch_loss/batch_count), (time.time() - start)))
        if args.evaluate_during_training:
            if epoch == (args.n_epochs - 1):
                print('EVALUATION AFTER FINAL EPOCH')
                actual_sentences, predicted_sentences, evaluation_accuracy, F_measure, AP, fig, classification = evaluate(model, val_data, model_config, other_config, loss_criterion=calculate_loss, final_eval=True)

                with open(os.path.join(pltdir, 'eval_results.txt'), 'w') as h:
                    h.write('Validation Classification Matrix \n{}\n\n'.format(classification))
                    h.write('Average Precision: {:.3f}\n'.format(AP))
                    h.write('F1 score: {:.3f}'.format(F_measure))
                    h.close()

                torch.save(model.state_dict(), os.path.join(pltdir, os.path.basename(args.args.outputdir)+'.pth'))
                print('Evaluation accuracy: {:.3f}, F1 score: {:.3f}, Average precision: {:.3f}'.format(evaluation_accuracy*100, F_measure*100, AP*100))
                print('Time taken: {}s'.format(time.time() - start))
            else:
                evaluation_accuracy, F_measure, AP, classification, fig = evaluate(model, val_data, model_config, other_config, loss_criterion=calculate_loss, final_eval=False)
                print('Evaluation accuracy: {:.3f}, F1 score: {:.3f}, Average precision: {:.3f}'.format(evaluation_accuracy*100, F_measure*100, AP*100))
                print('Time taken: {:.1f}s'.format(time.time() - start))
            F1_scores.append(F_measure)
            
        epoch_count += 1

    #visulaize training
    utils.visualize_model(average_accuracy, average_loss, os.path.basename(args.outputdir), pltdir)
    print('Total time taken to train: {}s'.format(np.round(time_taken, 2)))

    return model, calculate_loss


def evaluate(model, test_data, model_config, other_config, loss_criterion=None, final_eval=False):
    P,T,probs = [],[],[]
    model.eval()
    with torch.no_grad():
        acc, batch_count, loss = 0, 0, 0
        actual_sentences, predicted_sentences = [], []
        for sent, tags in test_data:
            sentence_len = len(tags)
            sentence, pos = sent
            sentence = sentence.to(dp.device)
            pos = pos.to(dp.device)
            tags = tags.to(dp.device)
            hidden = model.hidden

            if 'crf' in model_config and model_config['crf'] == True:
                l, feats = model(sentence, pos, hidden, tags)
                path_score, decoded_best_path = model.decode(feats)
                predicted = [i.item() for i in decoded_best_path]
            else:
                model_output = model(sentence, pos, hidden, tags)
                topv, topi = model_output.topk(1)
                predicted = [i.item() for i in topi]
                l = torch.tensor(0) if not loss_criterion else loss_criterion(model_output, tags.view(tags.size(0)))
                proba_pred = topv.to('cpu').numpy()
                for i in proba_pred[:, 0]:
                    probs.append(i)

            target = [i.item() for i in tags]
            target_sentence, predicted_sentence = utils.extract_predictions_from_model_output(target, predicted, other_config['index2tag'])
            actual_sentences.append(target_sentence)
            predicted_sentences.append(predicted_sentence)

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
        classes = [v for k,v in other_config['index2tag'].items()]
        F_measure, AP, fig, classification = utils.metrics_score(target=T, predicted=P, cls=classes, prfs=True, probs=probs)
        if final_eval:
            return actual_sentences, predicted_sentences, evaluation_accuracy, F_measure, AP, fig, classification
    return evaluation_accuracy, F_measure, AP, fig, classification

def eval_dur_train(args, mode):
    batching_configurations_eval = ut.load_data(args, mode=mode)
    eval_data = ut.batch_up_sets(args=args, batch_config=batching_configurations_eval, mode=mode)
    return eval_data

def main():
    print('Current file location :', os.path.abspath(os.path.curdir))

    data_dir = '../ebm-data/stanford/'
    output_dir = '../output/H_O_T'
    model_name = 'default_H_O_T'
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
    par.add_argument("--train_dev_test", default='train', type=str, required=True, help="Turn on or off training, evaluation and prediction phases")
    par.add_argument("--pretrained_embeddings", default='../word_vecs/bionlp_wordvec/pubmed_1.pickle', type=str, help="source of pretrained embeddings")
    par.add_argument("--use_pretrained_embeddings", action='store_true', help="use or ignore pretrained embeddings")
    par.add_argument("--use_bert_features", action='store_true', help="use or ignore pretrained embeddings")
    par.add_argument("--crf_layer", action='store_true', help="use or ignore crf_layer")
    par.add_argument("--neural", default='lstm', type=str, help ="LSTM or GRU or RNN")
    par.add_argument("--train_val_split", default=None, help="split the training data into train and validation sets")
    par.add_argument("--evaluate_during_training", action="store_true", help="Whether to run evaluation during training at each logging step.")
    par.add_argument("--sampling_percentage", default=None, type=int, help="percentage to use when sampling down the majority calss labels")
    par.add_argument("--sampling_technique", default='under', type=str, help="sampling strategy")

    args = par.parse_args()
    # create storacge location
    args.outputdir = utils.create_directories_per_series_des(args.outputdir)
    with open(os.path.join(args.outputdir, 'configuration.txt'), 'w') as cf:
        for k, v in vars(args).items():
            cf.write('{}: {}\n'.format(k, v))
        cf.close()

    if args.use_pretrained_embeddings and args.use_bert_features:
        print("You are using pretrained features and particularly bert features")
        if not os.path.basename(os.path.abspath(args.pretrained_embeddings)).lower().__contains__('bert'):
            raise ValueError("Double check that you're passing bert pickle object file, use argument --pretrained_embeddings")
        else:
            print("Right bert features passed")
    elif args.use_pretrained_embeddings:
        print("You are using pretrained pubmed vectors")
    else:
        print("You are using default embeddings")

    #load data
    modes = args.train_dev_test.split()
    
    if 'train' in modes:
        model_configurations, batching_configurations_train, other_configurations = ut.load_data(args, mode=modes[0])
        train_data, label_weights = ut.batch_up_sets(args=args, batch_config=batching_configurations_train, other_config=other_configurations, mode=modes[0])
        model = NNModel(model_configurations).to(dp.device)
        # initialize weights
        for param, values in model.named_parameters():
            if param.__contains__('weight'):
                torch.nn.init.xavier_normal_(values)
            elif param.__contains__('bias'):
                torch.nn.init.constant_(values, 0.0)
        # load_model and beging training it on the loaded data
        print('\nTRAINING BEGINS\n')
        best_model, calculate_loss = train(model=model,
                                           train_data=train_data,
                                           val_data= None if not args.evaluate_during_training else eval_dur_train(args, modes[1]),
                                           model_config=model_configurations,
                                           other_config=other_configurations,
                                           pltdir=args.outputdir,
                                           args=args,
                                           label_weights=label_weights,
                                           balanced_loss=True,
                                           loss_method=None,
                                           gamma=None)

    if 'eval' in modes:
        modes[1] = 'dev'
        print('\nEVALUATION BEGINS\n')
        batching_configurations_eval = ut.load_data(args, mode=modes[1])
        eval_data = ut.batch_up_sets(args=args, batch_config=batching_configurations_eval, mode=modes[1])

        actual_sentences, predicted_sentences, evaluation_accuracy, F_measure, AP, fig, classification = evaluate(model=best_model,
                                                                                                                  test_data=eval_data,
                                                                                                                  model_config=model_configurations,
                                                                                                                  other_config=other_configurations,
                                                                                                                  loss_criterion=calculate_loss,
                                                                                                                  final_eval=True)

        with open(os.path.join(args.outputdir, 'eval_results.txt'), 'w') as h:
            h.write('Validation Classification Matrix \n{}\n\n'.format(classification))
            h.write('Average Precision: {:.3f}\n'.format(AP))
            h.write('F1 score: {:.3f}'.format(F_measure))
            h.close()

        torch.save(best_model.state_dict(), os.path.join(args.outputdir, model_name + '.pth'))
        print('Evaluation accuracy: {:.3f}, F1 score: {:.3f}, Average precision: {:.3f}'.format(evaluation_accuracy * 100, F_measure * 100, AP * 100))

    if 'test' in modes:
        batching_configurations_test = ut.load_data(args, mode=modes[2])
        test_data = ut.batch_up_sets(args=args, batch_config=batching_configurations_test, mode=modes[2])
        actual_sentences, predicted_sentences, evaluation_accuracy, F_measure, AP, fig, classification = evaluate(model=model,
                                                                                                                  test_data=eval_data,
                                                                                                                  model_config=model_configurations,
                                                                                                                  other_config=other_configurations,
                                                                                                                  loss_criterion=calculate_loss,
                                                                                                                  final_eval=True)

        with open(os.path.join(args.outputdir, 'decoded.out'), 'w') as d, open(os.path.join(args.outputdir, 'test_results.txt'), 'w') as h, open(args.test_data_file, 'r') as q:
            h.write('Test Classification Matrix \n{}\n\n'.format(classification))
            h.write('Average Precision: {:.3f}\n'.format(AP))
            h.write('F1 score: {:.3f}'.format(F_measure))
            test_instances = q.readlines()
            f = [i.split()[0] if i != '\n' else i for i in test_instances]
            k, test_instances_list = '', []
            for i in f:
                if i != '\n':
                    k += ' {}'.format(i)
                elif i == '\n':
                    if k:
                        test_instances_list.append(k.strip())
                    k = ''

            for t, a, p in zip(test_instances_list, actual_sentences, predicted_sentences):
                for t_s, a_s, p_s in zip(t.split(), a.split(), p.split()):
                    d.write('{} {} {}\n'.format(t_s.strip(), a_s.strip(), p_s.strip()))
                d.write('\n')
            h.close()
            d.close()
            q.close()
        print('Test accuracy: {}, F1 score: {:.3f}, Average precision: {:.3f}'.format(evaluation_accuracy, F_measure, AP))

if __name__=='__main__':
    main()



