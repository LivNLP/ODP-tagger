# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 16/09/19 
# @Contact: michealabaho265@gmail.com

import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('../'))
import data_prep as dp
from seq_models.biLSTM import BiLstm_model
import numpy as np
import helper_functions as utils
import pickle
import time
#from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt

def train(model, inp_data, index2tag, patience, learning_rate, n_epochs, batch_size):

    calculate_loss = nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_accuracy = 0

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    average_accuracy, average_loss = [], []

    for epoch in range(n_epochs):
        batch_acc,batch_loss = 0,0
        batch_count = 0
        for batches in utils.batch_generator(inp_data, batch_size):
            acc, loss = 0,0
            for batch in batches:
                model.zero_grad()
                model.hidden = model.init_hidden()
                sentence, tags = batch[0].to(dp.device), batch[1].to(dp.device)
                tag_scores = model(sentence)
                topv, topi = tag_scores.topk(1)
                predicted = [i.item() for i in topi]
                target = [i.item() for i in tags]

                a = utils.token_level_accuracy(target, predicted)
                acc += a
                l = calculate_loss(tag_scores, tags.view(tags.size(0)))
                loss += np.round(l.item(), 4)
                #print('accuracy {} Loss {}'.format(a, np.round(l.item(), 4)))

                l.backward()
                optimizer.step()

            print('Batch Average accuracy:- {} and Average loss:- {}'.format((acc / batch_size), (loss / batch_size)))

            batch_acc += acc/batch_size
            batch_loss += loss/batch_size
            batch_count += 1

        epoch_loss += batch_loss/batch_count
        epoch_accuracy += batch_acc/batch_count
        average_accuracy.append((batch_acc / batch_count))
        average_loss.append((batch_loss / batch_count))

        print('epoch acc {} epoch loss {}'.format((batch_acc/batch_count), (batch_loss/batch_count)))
        if epoch % 5 == 0:
            print('Accuracy: {}, Loss: {}'.format(float(epoch_accuracy/5), float(epoch_loss/5)))
            batch_acc = 0
            batch_loss = 0


    return model, average_accuracy, average_loss


def evaluate(model, test_data, data_dir, index2tag):
    P,T,probs = [],[],[]
    with torch.no_grad():
        with open(os.path.join(data_dir, 'decoded.out'), 'w') as d, open(os.path.join(data_dir, 'baseline_model_results.txt'), 'w') as h:
            acc, acc_count = 0, 0
            for sentence, tags in test_data:
                sentence_len = len(tags)

                model_output = model(sentence)

                topv, topi = model_output.topk(1)
                target = [i.item() for i in topi]
                predicted = [i.item() for i in tags]

                target_sentence, predicted_sentence = utils.extract_predictions_from_model_output(target, predicted, index2tag)

                proba_pred = topv.to('cpu').numpy()

                for i in proba_pred[:,0]:
                    probs.append(i)

                for u,v in zip(target_sentence.split(), predicted_sentence.split()):
                    d.write('{} {}'.format(u.strip(), v.strip()))
                    d.write('\n')
                d.write('\n')

                acc += utils.token_level_accuracy(target, predicted)
                acc_count += 1
                P.append(predicted_sentence.split())
                T.append(target_sentence.split())


            evaluation_accuracy = float(acc/acc_count)
            print('Evaluation accuracy {}'.format(evaluation_accuracy))
            P = [i for j in P for i in j]
            T = [i for j in T for i in j]
            classes = [v for k,v in index2tag.items()]
            F_measure, AP, fig, classification = utils.metrics_score(target=T, predicted=P, cls=classes, prfs=True, probs=probs)
            h.write('Classification Matrix \n {} \n'.format(classification))
            h.write('Average Precision: {} \n'.format(AP))
            d.close()
            h.close()
            print(classification)
    return evaluation_accuracy, classification, fig


def visualize_model(accuracy, average_loss, model_name, plt_dir):
    # visualize the training
    x = range(0, len(accuracy))
    fig, (acc,lss) = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)
    acc.plot(x, accuracy)
    acc.set_title('Token level Accuracy')
    lss.plot(x, average_loss)
    lss.set_title('Average Loss')
    plt.legend()
    fig.savefig(os.path.join(plt_dir, '{}'.format(model_name)))

if __name__=='__main__':
    print(os.path.abspath(os.path.curdir))
    #load data
    file_path = '../../pico-outcome-prediction/corrected_outcomes/BIO-data/stanford'
    train_data_file = os.path.join(file_path, 'train_stanford_ebm.bmes')
    #test_data_file = os.path.join(file_path, 'Test_stanford_ebm.bmes')

    percentage_split = 0.8
    word_map, word_count, index2word, num_words = dp.create_vocabularly(train_data_file)
    line_pairs, outputs = dp.readwordTag(train_data_file)
    tag_map = dp.create_tag_map(outputs)
    index2tag = dict((v, k) for k, v in tag_map.items())
    train_input, train_target, max_length = dp.inputAndOutput(line_pairs, word_map, tag_map)
    data = list(zip(train_input, train_target))

    #shuffle the data
    np.random.shuffle(data)
    train_indices_len = int(np.round(percentage_split*len(data)))
    train_data = data[:train_indices_len]
    test_data = data[train_indices_len:]

    embedding_dim = 150
    hidden_dim = 256

    #initial_model
    bilstm = BiLstm_model(embedding_dim=embedding_dim, hidden_size=hidden_dim, vocab_size=num_words, output_size=len(tag_map), dropout_rate=0.2).to(dp.device)

    #training with glove
    embeddings_file = open('../word_vecs/glove/glove_1.pickle', 'rb')
    glove_embeddings = pickle.load(embeddings_file)
    bilstm_glove = BiLstm_model(embedding_dim=embedding_dim, hidden_size=hidden_dim, vocab_size=num_words, output_size=len(tag_map), dropout_rate=0.2, weights_tensor=glove_embeddings, use_pretrained_embeddings=True).to(dp.device)

    model_name = "bilstm_glove"
    plot_dir = "plots"

    start = time.time
    trained_model, accuracy, average_loss = train(model=bilstm_glove,
                                                  inp_data=train_data,
                                                  index2tag=index2tag,
                                                  n_epochs=100,
                                                  learning_rate=0.1,
                                                  patience=12,
                                                  batch_size=250)

    #visualize_training
    pltdir = utils.create_directories_per_series_des(plot_dir)
    visualize_model(accuracy, average_loss, model_name, pltdir)

    #evaluation
    evaluation_accuracy, classification, fig = evaluate(trained_model, test_data, pltdir, index2tag)
    fig.savefig(os.path.join(pltdir, 'metrics.png'))
    torch.save(trained_model.state_dict(), os.path.join(pltdir, 'bilstm_glove.pth'))

    print('Duration {}'.format(time.time() - start))