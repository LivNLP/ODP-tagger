import torch
import torch.optim as optim
import torch.nn as nn
import data_prep as dp
from lstm import Lstm_model
import numpy as np
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
import os

def train(inp_data, index2tag, embedding_dim, hidden_dim, vocab, output_size, dropout_rate, patience, learning_rate, n_epochs, model_name=''):
    index2tag = dict((v, k) for k, v in vocab.items())
    if str(model_name).lower() == 'lstm':
        vocab_size = len(vocab)
        model = Lstm_model(embedding_dim, hidden_dim, vocab_size, output_size, dropout_rate).to(dp.device)
        calculate_loss = nn.CrossEntropyLoss()
        average_loss = []
        loss_per_sequence = []
        acc = []
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        with torch.no_grad():
            inputs = dp.prepare_tensor(inp_data[0][0])
            tag_scores = model(inputs)
            #print(tag_scores)

        for epoch in range(n_epochs):
            s, t = 0, 0
            for sentence, tags in inp_data:
                model.zero_grad()
                model.hidden = model.init_hidden()

                sentence_in = dp.prepare_tensor(sentence)
                targets = dp.prepare_tensor(tags)

                tag_scores = model(sentence_in)
                target, predicted_sentence_list = extract_predictions_from_model_output(tag_scores, tags, index2tag)

                loss = calculate_loss(tag_scores, targets)
                loss_per_sequence.append(loss.item())

                loss.backward()
                optimizer.step()

                sim, total = token_level_accuracy(target, predicted_sentence_list)
                s += sim
                t += total

            acc.append(float(s/t))
            #print(float(s/t))
            if loss_per_sequence:
                average_loss.append(np.mean([i for i in loss_per_sequence]))
            loss_per_sequence.clear()

        with torch.no_grad():
            inputs = dp.prepare_tensor(inp_data[0][0])
            tag_scores = model(inputs)
            #print(tag_scores)

    return model, acc, average_loss


def evaluate(model, test_data, tag_map, data_dir):
    P,T = [],[]
    index2tag = dict((v,k) for k,v in tag_map.items())
    file_dir = os.path.dirname(data_dir)
    decoded_file_name = os.path.basename(data_dir).split('.')[0]
    with torch.no_grad():
        with open(os.path.join(file_dir, '{}.out'.format(decoded_file_name)), 'w') as d, open(os.path.join(file_dir, 'baseline_model_results.txt'), 'w') as h:
            s,t = 0,0
            for sentence, tags in test_data:
                sentence_len = len(tags)
                sentence_in = dp.prepare_tensor(sentence)
                model_output = model(sentence_in)

                target, predicted_sentence_list = extract_predictions_from_model_output(model_output, tags, index2tag)

                for u,v in zip(target, predicted_sentence_list):
                    d.write('{} {}'.format(u.strip(), v.strip()))
                    d.write('\n')
                d.write('\n')

                sim, total = token_level_accuracy(target, predicted_sentence_list)
                s += sim
                t += total
                P.append(predicted_sentence_list)
                T.append(target)

            acc = float(s/t)
            P = [i for j in P for i in j]
            T = [i for j in T for i in j]
            metrics = metrics_score(T, P, tag_map.keys())
            print(metrics)
            h.write(metrics)
            d.close()
            h.close()
    return acc, metrics

def extract_predictions_from_model_output(model_output, true_tags, index2tag):
    topv, topi = model_output.data.topk(1)
    predicted_sequence = [i.item() for i in topi]
    predicted_sentence_list = [index2tag[i] for i in predicted_sequence]
    target = [index2tag[i] for i in true_tags.tolist()]
    return target, predicted_sentence_list

def token_level_accuracy(target, predicted):
    sim, total = 0, 0
    for i,j in zip(target, predicted):
        if i == j:
            sim += 1
        total += 1
    return sim, total

def metrics_score(target, predicted, cls, prfs=False, classification=False):
    if prfs:
        p,r,f,s = precision_recall_fscore_support(target, predicted)
        return p,r,f,s
    return classification_report(target, predicted, target_names=cls)

def visualize_model(accuracy, average_loss, model_name, plot_dir):
    # visualize the training
    x = range(0, len(accuracy))
    fig, (acc,lss) = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)
    acc.plot(x, accuracy, label='Token level Accuracy')
    lss.plot(x, average_loss, label='Average Loss')
    fig.savefig(os.path.join(os.path.abspath(plot_dir), '{}'.format(model_name)))

if __name__=='__main__':
    #load data
    file_path = 'ebm-data'
    train_data_file = os.path.join(file_path, 'train_ebm.bmes')
    test_data_file = os.path.join(file_path, 'test_ebm.bmes')
    word_map, word_count, index2word = dp.create_vocabularly(train_data_file)

    line_pairs, outputs = dp.readwordTag(train_data_file)

    tag_map = dp.create_tag_map(outputs)
    train_input, train_target, tr_lengths = dp.inputAndOutput(line_pairs, word_map, tag_map)
    train_data = list(zip(train_input, train_target))
    embedding_dim = 10
    hidden_dim = 64

    #training
    model_name = "lstm"
    plot_dir = "plots"
    index2tag = dict((v, k) for k, v in tag_map.items())
    print(index2tag)
    trained_model, accuracy, average_loss = train(train_data[:50],
                                                  embedding_dim=embedding_dim,
                                                  hidden_dim=hidden_dim,
                                                  index2tag=index2tag,
                                                  vocab=word_map,
                                                  output_size=len(tag_map),
                                                  n_epochs=30,
                                                  dropout_rate=0.2,
                                                  learning_rate=0.1,
                                                  patience=12,
                                                  model_name=model_name)

    #visualize_training
    visualize_model(accuracy, average_loss, model_name, plot_dir)

    #evaluation
    line_pairs2, outputs2 = dp.readwordTag(test_data_file)
    test_input, test_target, te_lengths, oov_words = dp.inputAndOutput(line_pairs2, word_map, tag_map, test=True)
    test_data = list(zip(test_input, test_target))
    a, m = evaluate(trained_model, test_data[:20], tag_map, test_data_file)
    print(a)
    print(m)