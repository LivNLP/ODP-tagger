import os
import sys
sys.path.append('../')
sys.path.append('../../BNER-tagger')
import ebm_nlp_demo as e
import data_prep as dp
import spacy
import numpy as np

medpost = dp.spacyModel()
'''
Extracting data from from the EBM-NLP-corpus
'''
def extract_pico_elements(pico_element, model_phase):
    pico_element = pico_element.lower().strip()
    model_phase = model_phase.lower().strip()
    turker, ebm_extract = e.read_anns('starting_spans', pico_element, \
                                      ann_type='aggregated',
                                      model_phase=model_phase if model_phase=='train' else model_phase+'/gold')

    pico_element_dir = os.path.join(os.path.curdir, pico_element)
    if not os.path.exists(pico_element_dir):
        os.makedirs(pico_element_dir)
    i = 0
    with open('{}.txt'.format(model_phase), 'w') as train, open(os.path.join(pico_element_dir, '{}.txt'.format(model_phase)), 'w') as train_:
        for pmid, doc in ebm_extract.items():
            annotations = doc.anns['AGGREGATED']
            tokens = doc.tokens
            token_count = range(len(tokens))
            x = 0
            element_anns = []
            for count, token, label in zip(token_count, tokens, annotations):
                if x == count:
                    label = doc.decoder[label]
                    if label.lower() == 'no label':
                        train.write(token + ' ' + 'O' + '\n')
                        element_anns.append((token, 'O'))
                    else:
                        train.write(token + ' ' + 'B-'+label + '\n')
                        element_anns.append((token, 'B-'+label))
                        for tok,ann in zip(tokens[count+1:], annotations[count+1:]):
                            if doc.decoder[ann].lower() != 'no label':
                                train.write(tok + ' ' + 'I-'+doc.decoder[ann] + '\n')
                                element_anns.append((tok, 'I-'+doc.decoder[ann]))
                                x += 1
                            else:
                                break
                    x += 1
            train.write('\n')

            segmented_abstracts = split_data(element_anns)
            for sentence in segmented_abstracts:
                for tok, label, pos, char in zip(*sentence):
                    train_.write(tok + ' ' + label + '\n')
                train_.write('\n')
            train_.write('\n')
            i += 1

    print('{} dataset length {}'.format(model_phase, i))


def split_data(tokens_anns_list, transformer=False):
    num_instances = 0
    num_abstracts = 0
    segmented_abstract, sentences = [], []
    tokens, anns = list(zip(*tokens_anns_list))
    tokens, anns = list(tokens), list(anns)
    abstract = ' '.join(tokens)
    doc_abstract = medpost(abstract)

    for i, sent in enumerate(doc_abstract.sents):
        #
        # in addition to a word, include two other features, part of speech and the whether or not the first character of the word is upper case or not
        #
        split_sentence = str(sent).split()
        if transformer:
            pos = [i.pos_ for i in medpost(str(sent))]
            char_case_feature = [i[0].isupper() for i in split_sentence]
            char_case_feature = ['T' if i == True else 'F' for i in char_case_feature]
        else:
            pos = char_case_feature = [np.NAN]*len(split_sentence)

        tags = anns[:len(split_sentence)]

        for j in [split_sentence, tags, pos, char_case_feature]:
            if str(j) != 'nan':
                sentences.append(j)
        if sentences:
            segmented_abstract.append([i for i in sentences])
        sentences.clear()
        anns = anns[len(split_sentence):]

    return segmented_abstract

args = sys.argv
if len(args) == 3:
    extract_pico_elements(args[1], args[2])
else:
    raise ValueError("Check your arguments, Either you haven not passed the model phase argument or you have more than required arguments")
