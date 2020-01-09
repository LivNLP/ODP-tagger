# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 11/12/19 
# @Contact: michealabaho265@gmail.com
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import re
import numpy as np
import torch.nn as nn
import torch
import json
import pprint
import pickle
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import sys
sys.path.append('../../transformers/')
from examples.utils_ner import get_labels, read_examples_from_file
from transformers import BertModel, BertConfig, BertTokenizer
from examples.run_ner import load_and_cache_examples


MODEL_CLASSES = {"bert": (BertConfig, BertModel, BertTokenizer)}

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))

    return features

def load_and_cache_examples(args, tokenizer, labels, mode):
    examples = read_examples_from_file(args.data_dir, mode=mode)
    features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(args.model_type in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(args.model_type in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--file_name", default="train", type=str,
                        help="e.g dev for development dataset, train for training dataset etc")
    parser.add_argument("--emb_dim", default=384, type=int,
                        help="feture dimension")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        pass
    else:
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, args.per_gpu_eval_batch_size, bool(args.local_rank != -1)))

    # Prepare ebm-task
    labels = get_labels(args.labels)
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    #load dataset
    eval_data = load_and_cache_examples(args, tokenizer, labels, mode=args.file_name)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu access
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #create a list of the example instances directly from the file to use for reconciling the tokenization differences between BERT and the health tagger model
    with open(os.path.join(args.data_dir, args.file_name+'.txt'), 'r') as rf:
        rf_lines = rf.readlines()
        file_content, file_instance = [], []
        for v in rf_lines:
            if v != '\n':
                file_instance.append(v.split()[0].strip())
            else:
                if file_instance:
                    file_content.append(file_instance.copy())
                file_instance.clear()

    #fetch the sequence outputs as features from the Bert model encoder and construct a json with each token with their corresponding feature vector
    count = 0
    Instances = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert","xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

            outputs, _ = model(**inputs)
            print('--------------------------------------------------------------NEW BATCH--------------------------------------------------------------')
            for b, output in enumerate(outputs):
                token_ids = inputs["input_ids"][b].cpu().numpy().tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                t = []
                d = 0
                x = tokenizer.convert_tokens_to_string(tokens)
                #print(' '.join(i for i in x.split() if i not in ['[CLS]', '[PAD]', '[SEP]']))
                #print(' '.join(file_content[count]))
                m = 0
                #print(tokens)
                for c, tok in enumerate(tokens):
                    if c == d:
                        if tok not in ['[CLS]', '[PAD]', '[SEP]']:
                            #print(tok)
                            word = tok
                            layers_output = np.array([round(i.item(), 6) for i in output[c]])
                            if file_content[count][m] == word:
                                #print('Immediate', word, file_content[count][m])
                                d += 1
                                m += 1
                                t.append((word, layers_output))
                            else:
                                for k in range(c+1, len(tokens)):
                                    if tokens[k] not in ['[CLS]', '[PAD]', '[SEP]']:
                                        additional_toks = re.sub('^(##)', '', tokens[k])
                                        word += additional_toks.strip()
                                        #print(word)
                                        layers_output = np.add(layers_output, np.array([round(i.item(), 6) for i in output[k]]))
                                        if file_content[count][m] == word:
                                            #print(word, file_content[count][m], k, d)
                                            m += 1
                                            curr_loc = k
                                            t.append((word, layers_output))
                                            break
                                #print('current location', curr_loc)
                                d = curr_loc + 1
                        else:
                            d += 1
                    else:
                        pass
                Instances[count+1] = t
                count += 1
    with open(os.path.join(os.path.abspath(args.output_dir), 'bert_embeddings_{}.pickle'.format(args.file_name)), 'wb') as writer:
        pickle.dump(Instances, writer)

if __name__ == "__main__":
    main()