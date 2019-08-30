import torch
import torch.nn as nn
import sys
sys.path.append('/users/phd/micheala/Documents/Github/Health-outcome-tagger/')
import outcome_model as model
import torch.optim as optim
import data_prep as data_loader
import helper_functions as utils
import time
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_outcome_token, end_outcome_token = 0, 1

def train(encoder, decoder, epochs, input_words, output_tags, max_len, learning_rate, print_every, batch_size, use_teacher_forcing=False):
    losses = []
    print_loss_total = 0
    plot_loss_total = 0

    #define backprop strategy
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    loss_criterion = nn.NLLLoss()
    start = time.time()
    epoch_loss = 0
    for epoch in range(epochs):
        training_tensors = list(zip(input_words, output_tags))
        epoch_batch_loss,epoch_batch_acc  = 0, 0
        for batch_pairs in utils.batch_generator(training_tensors, batch_size):
            batch_loss, batch_acc  = 0, 0
            for batch_tensor in batch_pairs:
                input_tensor = batch_tensor[0].to(device)
                target_tensor = batch_tensor[1].to(device)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                input_len, target_len = input_tensor.size(0), target_tensor.size(0)

                encoder_hidden = encoder.initHidden()
                encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

                loss = 0

                for e in range(input_len):
                    encoder_output, out_encoder_hidden = encoder(input_tensor[e], encoder_hidden)
                    encoder_outputs[e] = encoder_output[0, 0]

                decoder_hidden = out_encoder_hidden
                decoder_input = torch.tensor([[start_outcome_token]], device=device)
                predicted_tags = []
                if use_teacher_forcing:
                    for d in range(target_len):
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                        loss += loss_criterion(decoder_output, target_tensor[d])
                        decoder_input = target_tensor[d]

                else:
                    for d in range(target_len):
                        decoder_output, out_decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach()
                        loss += loss_criterion(decoder_output, target_tensor[d])
                        if decoder_input.item() == end_outcome_token:
                            break
                        else:
                            predicted_tags.append(topi.item())
                loss.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()

                #loss
                batch_loss += (loss.item()/target_len)

                #accuracy
                true_tags = [i.item() for i in target_tensor][:-1]
                predicted_tags = predicted_tags[1:]
                sim, tot = utils.token_level_accuracy(true_tags, predicted_tags)
                batch_acc += sim/tot
            epoch_batch_loss += (batch_loss/len(batch_pairs))
            epoch_batch_acc = (batch_acc/len(batch_pairs))
            print("AVERAGE BATCH_LOSS {} AVERAGE TOKEN LEVEL ACCURACY {}".format((batch_loss/len(batch_pairs)), (epoch_batch_acc)))
        epoch_loss += (epoch_batch_loss/batch_size)
        losses.append(epoch_batch_loss/batch_size)
        print('EPOCH_LOSS {}'.format(epoch_batch_loss/batch_size))
        if (epoch+1) % print_every == 0:
            now = time.time()
            print('Average Loss after {} epochs is {},  Duration - {:.4f}'.format((epoch+1), (epoch_loss/print_every), (now-start)))
            epoch_loss = 0


def evaluate(encoder, decoder, input_words, embedding_dim, hidden_size, output_tags, max_len, word_map, index2word):
    input_length = input_words.size(0)
    encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)
    loss_criterion = nn.NLLLoss()

    loss = 0

    for u in range(input_length):
        encoded_output, encoder_hidden = encoder(input_words[u], encoder_hidden)
        encoder_outputs[u] += encoded_output[0, 0]

    decoder_hidden = encoder_hidden
    decoded_tags = []

    loss_criterion = nn.NLLLoss()

    decoder_input = torch.tensor([[start_outcome_token]])
    for v in range(max_len):
        decoder_output, dec_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        decoded_tags.append(index2word[topi.item()])
        if v < input_length:
            loss += loss_criterion(decoder_output, output_tags[v])
            decoder_input = topi.squeeze().detach()
        else:
            break
    return decoded_tags, loss.item()

def load_and_process(file_path, evaluation=False):
    word_map, word_count, index2word = data_loader.create_vocabularly(file_path)
    line_pairs, outputs = data_loader.readwordTag(file_path)
    tag_map = data_loader.create_tag_map(outputs)
    input_words, output_tags, max_len = data_loader.inputAndOutput(line_pairs, word_map, tag_map)
    if evaluation:
        return word_map, index2word, line_pairs, outputs, input_words, output_tags, max_len
    else:
        return line_pairs, word_map, index2word, outputs, input_words, output_tags, max_len


if __name__=='__main__':
    #training
    file_path = 'ebm-data/train_ebm.bmes'
    linepairs, word_map, index2word, outputs, input_words, output_tags, max_len = load_and_process(file_path)

    embedding_dim = 50
    hidden_size = 100
    dropout=0.1

    encoder = model.outcomeEncoder(embedding_dim=embedding_dim,
                                   hidden_size=hidden_size,
                                   vocab_size=len(word_map)).to(device)

    decoder = model.outcomeDecoder(hidden_size=hidden_size,
                                   output_size=len(outputs),
                                   dropout=dropout,
                                   n_layers=1).to(device)

    attndecoder = model.outcomeAttnDecoder(hidden_size=hidden_size,
                                           output_size=len(outputs),
                                           dropout=dropout,
                                           n_layers=1,
                                           max_length=max_len).to(device)

    train(encoder=encoder,
          decoder=attndecoder,
          input_words=input_words,
          output_tags=output_tags,
          max_len=max_len,
          epochs=100,
          batch_size=375,
          learning_rate=0.001,
          print_every=5)


    # #testing
    #
    # w, i, l, o, input_test_words, output_test_tags, max_l = load_and_process("ebm-data/test_ebm.bmes", evaluation=True)
    # testing_loss = []
    # with open('decoded.out', 'w') as dec_out:
    #     for x,y,z in zip(input_test_words, output_test_tags, l):
    #         predicted, loss = evaluate(encoder=encoder,
    #                  decoder=decoder,
    #                  input_words=x,
    #                  embedding_dim=embedding_dim,
    #                  hidden_size=hidden_size,
    #                  output_tags=y,
    #                  max_len=max_len,
    #                  word_map=word_map,
    #                  index2word=index2word)
    #
    #         dec_out.writelines(predicted)
    #         dec_out.writelines(z[0])
    #         dec_out.writelines('\n')
    #         testing_loss.append(loss/len(i))
    #     dec_out.close()
    #
    # print(np.mean(testing_loss))






