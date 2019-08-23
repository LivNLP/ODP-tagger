import torch
import torch.nn as nn
from seq_models import outcome_model as model
import torch.optim as optim
import data_prep as utils
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, learning_rate, print_every, use_teacher_forcing=False):
    file_path = 'ebm-data/train_ebm.bmes'
    word_map, word_count, index2word = utils.create_vocabularly(file_path)
    line_pairs, outputs = utils.readwordTag(file_path)
    encoded_outputs = utils.create_tag_map(outputs)
    input_words, output_tags, max_len = utils.inputAndOutput(line_pairs, word_map, encoded_outputs)

    encoder = model.outcomeEncoder(embedding_dim=50,
                                   hidden_size=100,
                                   vocab_size=len(word_map),
                                   n_layers=1).to(device)

    decoder = model.outcomeDecoder(hidden_size=100,
                                   output_size=len(outputs)).to(device)

    losses = []
    print_loss_total = 0
    plot_loss_total = 0

    #define backprop strategy
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    loss_criterion = nn.NLLLoss()

    for iter in range(epochs):
        start = time.time()
        input_tensor = input_words[iter]
        target_tensor = output_tags[iter]

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_len, target_len = input_tensor.size(0), target_tensor.size(0)

        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

        loss = 0

        for e in range(input_len):
            encoded_output, out_encoder_hidden = encoder(input_tensor[e], encoder_hidden)
            encoded_outputs[e] = encoded_output[0, 0]

        decoder_hidden = out_encoder_hidden

        if use_teacher_forcing:
            for d in range(target_len):
                decoder_output, decoder_hidden = decoder(target_tensor[d], decoder_hidden)
                loss += loss_criterion(decoder_output, target_tensor[d])

        else:
            decoder_input = target_tensor[0]
            for d in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += loss_criterion(decoder_output, target_tensor[d])

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        print_loss_total += loss

        #if (iter) % print_every == 0:
        now = time.time()
        print('Average Loss {} Duration - {:.4f}'.format((print_loss_total), (now-start)))

        return loss.item()/target_len

train(epochs=10, learning_rate=0.001, print_every=2)


