from docx import *
import xmltodict
import re
import spacy
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
import string
import spacy
import os
from pprint import pprint
from tabulate import tabulate
import data_prep as dp
import logging
from spacy.lang.en import English

core_areas = ['Physiological Clinical', 'Death', 'Life Impact', 'Resource use', 'Adverse effects']
COMET_LABELS = {\
    core_areas[0]:{\
        0:'Physiological/clinical'
        },
    core_areas[1]:{\
        1:'Mortality/survival'
        },
    core_areas[2]:{\
        25:'Physical_functioning',
        26:'Social_functioning',
        27:'Role_functioning',
        28:'Emotional_functioning/wellbeing',
        29:'Cognitive_functioning',
        30:'Global_quality_of_life',
        31:'Perceived_health_status',
        32:'Delivery_of_care',
        33:'Personal_circumstances'
        },
    core_areas[3]:{\
        34:'Economic',
        35:'Hospital',
        36:'Need_for_further_intervention',
        37:'Societal/carer_burden'
        },
    core_areas[4]:{
        38:'Adverse_events/effectsn'
    }
    }

logger = logging.getLogger(__name__)

def map_annotations(tag_number):
    label = 'No label'
    possible_labels = [v for y in COMET_LABELS for x,v in COMET_LABELS[y].items()]
    z_similar = []
    for area in COMET_LABELS:
        for tag, tag_label in COMET_LABELS[area].items():
            if type(tag_number) == str and str(tag) == tag_number.strip():
                tag_label_list = [i if i == tag_label else 0 for i in possible_labels]
                label = tag_label_list
            elif type(tag_number) == list:
                for i in tag_number:
                    if str(tag) == i.strip():
                        z_similar.append(tag_label)
                    if len(z_similar) == len(tag_number):
                        break
                tag_label_list = [i if i in z_similar else 0 for i in possible_labels]
                label = tag_label_list
    return label


def extract_annotations(files, path):
    #pick only .docx files and create three seperate data frames outcomes, tools and highlighted words
    files = [i for i in files if i.endswith('.docx')]
    comet_data = {}
    comet_outcomes_frame = pd.DataFrame()
    highlights_frame = pd.DataFrame()
    tools_frame = pd.DataFrame()
    spacy_model = dp.spacyModel()
    sentencizer = English()
    sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))

    with open(os.path.join(path, '31-40-dataset.txt'), 'w') as comet:
        for file in files:
            doc = Document(os.path.abspath(file))
            doc_content = doc.paragraphs #extract content from the word document
            dataset = []
            doc_tag_list = []
            highlights = []
            tools = []
            if (os.path.basename(file)) == 'RCT abstracts (31-40) FINAL v2.docx':
                print(os.path.basename(file))
                #comet.write(os.path.basename(file)+'\n')
                for s,t in enumerate(doc_content):
                    #abs_text = t.text
                    text_out = t.text
                    if len(text_out.split()) > 0:
                        doc_sents = sentencizer(text_out)
                        #fetch highlighted words
                        for v in fetch_highlights(t):
                            highlights.append(v)
                        #pick text annotated as tools
                        picked_tools = pickup_tools(text_out)
                        if len(picked_tools) > 0:
                            for tool in picked_tools:
                                tools.append(tool)

                        doc_sents = sentencizer(text_out)
                        for i, abs_text in enumerate(doc_sents.sents):
                            abs_text = abs_text.text
                            org_abs_text = abs_text
                            actual_abs_text = abs_text
                            tag_labels = []
                            #print(abs_text)
                            #look through lines in the paragrpahs, identifying outcomes annotated by <P #>.....</>
                            for ann in re.findall('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\/\>)', abs_text):
                                tag_labels.clear()
                                annot = ann[1]
                                #check nested outcomes outcomes <P #>..<P #>..<\>...</>, detected with closing tag <\>
                                if re.search('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\\\>)', annot):
                                    tag_no = re.findall(r'\d+', ann[0])
                                    phrase = re.search('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\\\>)', annot)
                                    sss = ' '.join(annot)
                                    tagged_items = []
                                    for tg in tag_no:
                                        annot_span = phrase.groups()[1].strip()
                                        doc_tag_list.append(('P {}'.format(tg), annot_span))
                                    annot = re.sub(re.escape(phrase.group().strip()), '', annot)
                                annotation = [i.strip() for i in ann]

                                #dealing with precise outcomes, no additional criteria required to extract outcomes from within openning and closing tags of an outcome i.e. no nested tags
                                if not re.search('\<P\s[\d\,\s]+\>', annotation[1]):
                                    #print(org_abs_text)
                                    tag_no = re.findall(r'\d+', annotation[0])
                                    phrase = annotation[1].strip(' ')
                                    phrasex = remove_tools_from_outcomes(phrase) #use for removing tools from outcomes
                                    sss = ' '.join(annotation)
                                    tagged_items = []
                                    #print(annotation)
                                    #case 1
                                    if len(tag_no) == 1:
                                        tag_label = map_annotations(tag_no[0])
                                        #print('case 1 {} {}'.format(tag_label, tag_no[0]))
                                        tagged_items, inner_phrase = case1_case2_case3_preprocess(phrase, tagged_items, 'U')
                                        doc_tag_list.append(('P {}'.format(tag_no[0]), phrase))

                                    #case 2
                                    elif len(tag_no) > 1:
                                        tag_label = map_annotations(tag_no)
                                        #print('case 2 {} {}'.format(tag_label, tag_no))
                                        tagged_items, inner_phrase = case1_case2_case3_preprocess(phrase, tagged_items, 'U')
                                        for tg in tag_no:
                                            doc_tag_list.append(('P {}'.format(tg), phrase))
                                    #print(tag_label)
                                    for _ in tagged_items:
                                        tag_labels.append(tag_label)
                                    # print(tag_labels)
                                    org_abs_text = org_abs_text.replace(sss, ' '+' '.join([str(i) for i in tagged_items])+' ', 1)
                                    #org_abs_text = re.sub(re.escape(sss), ' '+' '.join([str(i) for i in tagged_items])+' ', org_abs_text) #annotate with BIO tag
                                    actual_abs_text = de_annotate(actual_abs_text, annotation) #deannotation relace the xml annotation

                                #dealing with outcomes that either share and ending or a start, additional criteria to process these outcomes
                                elif re.search('^\(E\d+\)', annotation[1].strip()) or re.search('^\(S\d+\)', annotation[1].strip()):
                                    tags = re.findall('\s?\<P\s[\d\,\s]+\>\s?', annotation[1])
                                    tags.insert(0, annotation[0])
                                    tags_no = [re.findall(r'\d+', i) for i in tags]
                                    tags = [i.strip(' <>') for i in tags]

                                    x = int(annotation[1][2])
                                    annotation_ = annotation[1][5:]
                                    sss = annotation[0]+' '.join(annotation[1:])
                                    tagged_items = []

                                    #case 3
                                    if re.search('^\(E\d+\)', annotation[1].strip()):
                                        phrases = re.split('\s*\<P\s[\d\,\s]+\>\s*', annotation_)
                                        end = phrases[-1].split()
                                        print(annotation)
                                        #last_phrase = [i for i in end if i not in get_un_needed()]
                                        e_labels = []
                                        for i in range(len(phrases)):
                                            if i < len(phrases)-1:
                                                tagged_items, inner_phrase = case1_case2_case3_preprocess(phrases[i], tagged_items, 'E', x)
                                                inner_phrase = inner_phrase + ' ' + ' '.join(end[-(x):])
                                                inner_phrase = re.sub('\,','',inner_phrase)
                                                tag_label = capture_multi_labels(tags_no[i])
                                                for _ in range(len(tagged_items)):
                                                    e_labels.append(tag_label)
                                                print('inner_tags', [tag_label]*len(tagged_items))
                                                doc_tag_list.append((tags[i].strip(), inner_phrase))

                                        last_phrase = ' '.join(end)
                                        tagged_items, last_phrase = case1_case2_case3_preprocess(last_phrase, tagged_items, 'E', x, -1)
                                        last_tag_label = capture_multi_labels(tags_no[-1])
                                        print('last_tag', [last_tag_label] * len(tagged_items[len(e_labels) - 1:]))
                                        for _ in range(len(tagged_items[len(e_labels)-1:])):
                                            e_labels.append(tag_label)

                                        e_labels.append(last_tag_label)
                                        print(tagged_items)
                                        print(e_labels)
                                        for i in e_labels:
                                            tag_labels.append(i)

                                        doc_tag_list.append((tags[-1], last_phrase))
                                        org_abs_text = org_abs_text.replace(sss, ' '+' '.join([str(i) for i in tagged_items])+' ', 1)
                                            #re.sub(re.escape(sss), ' '+' '.join([str(i) for i in tagged_items])+' ', org_abs_text) #annotate with BIO tag
                                        actual_abs_text = de_annotate(actual_abs_text, annotation)  # deannotation relace the xml annotation
                                    #case 4
                                    elif re.search('^\(S\d+\)', annotation[1].strip()):
                                        phrases = re.split('\s*\<P\s[\d\,\s]+\>\s*', annotation_)
                                        start = phrases[0].split()
                                        tagged_items, inner_phrase = case1_case2_case3_preprocess(phrases[0], tagged_items, 'S', x)
                                        s_labels = []
                                        #print(annotation)
                                        for i in range(len(phrases)):
                                            if i > 0:
                                                last_phrase, end = ' ', re.split(' ', phrases[i])
                                                n = 0
                                                for h in range(len(end)):
                                                    if  n == h:
                                                        if end[h] in get_un_needed():
                                                            tagged_items.append('Seperator')
                                                        else:
                                                            if end[h].startswith('{'):
                                                                curly_braces_text = []
                                                                for j in range(h, len(end)):
                                                                    if not end[j].endswith('}'):
                                                                        curly_braces_text.append(end[j])
                                                                        tagged_items.append('X')
                                                                    else:
                                                                        curly_braces_text.append(end[j])
                                                                        tagged_items.append('X')
                                                                        n = j
                                                                        break
                                                                #print('Curly braces words:', ' '.join(curly_braces_text))
                                                            else:
                                                                if i == len(phrases)-1:
                                                                    if h == len(end) - 1:
                                                                        tagged_items.append('S{}-outcome'.format(x))
                                                                    else:
                                                                        tagged_items.append('I-outcome')
                                                                else:
                                                                    tagged_items.append('I-outcome')
                                                                last_phrase += end[h]
                                                        n += 1
                                                tag_label = capture_multi_labels(tags_no[i])
                                                s_labels.append(tag_label)
                                                last_phrase = ' '.join(start[:(x)]) + last_phrase
                                                doc_tag_list.append((tags[i], last_phrase))

                                        last_tag_label = capture_multi_labels(tags_no[0])
                                        s_labels.append(last_tag_label)
                                        #print(s_labels)
                                        for i in s_labels:
                                            tag_labels.append(i)

                                        doc_tag_list.append((tags[0], inner_phrase))
                                        org_abs_text = org_abs_text.replace(sss, ' '+' '.join([str(i) for i in tagged_items])+' ', 1)
                                        #org_abs_text = re.sub(re.escape(sss), ' '+' '.join([str(i) for i in tagged_items])+' ', org_abs_text) #annotate with BIO tag
                                        actual_abs_text = de_annotate(actual_abs_text, annotation)  # deannotation relace the xml annotation

                                #any other outcomes
                                else:
                                    print('Uncaught annotations for your attention', annotation, file)
                                    pass

                            actual_abs_text_split = re.split('\s+', actual_abs_text)
                            org_abs_text_split = re.split('\s+', org_abs_text)

                            print(i, tag_labels)

                            # print(org_abs_text)
                            # print(len(tag_labels) == len([i for i in org_abs_text_split if i.co]))
                            if len(actual_abs_text_split) == len(org_abs_text_split):
                                if not (actual_abs_text.__contains__('_PD.txt')) and not re.search('^(https:\/\/.)', actual_abs_text):
                                    docs_text, docs_pos, docs_tags = [], [], []
                                    # segment the abstracts into sentences
                                    for x, y in zip(actual_abs_text_split, org_abs_text_split):
                                        doc_x = [(u.text, u.pos_) for u in spacy_model(x)]
                                        doc_y = [y]*len(doc_x)
                                        for text_pos,label in zip(doc_x, doc_y):
                                            text, pos = text_pos
                                            text, pos, label = text.strip(), pos.strip(), label.strip()
                                            if text.endswith(']') and not text.startswith('['):
                                                text = remove(text, ['\]'])
                                            if re.search('(\[T)|(\{\[T)', text):
                                                text = remove(text, ['(\[T)', '\{\[T'])
                                                label = remove(label, ['(\[T)', '\{\[T'])
                                            if text:
                                                if re.search('(-outcome)$|^X$|^(Seperator)$', label):
                                                    text = remove(text=text, unwanted_patterns=['{', '}'])
                                                    docs_tags.append(label)
                                                else:
                                                    docs_tags.append('O')
                                                #print(text, pos, label)
                                                docs_text.append(text)
                                                docs_pos.append(pos)
                                    for te,ta,po in zip(docs_text, docs_tags, docs_pos):
                                        comet.write('{} {} {}\n'.format(te, ta, po))
                                    comet.write('\n')
                            else:
                                logger.info('OOPS pay attention to the below')
                                for u,v in zip(actual_abs_text_split, org_abs_text_split):
                                    print(u, v)
                                print(len(actual_abs_text_split), len(org_abs_text_split))
                                print('Before \n {} \n After \n {} \n {} \n {} \n'.format(actual_abs_text_split, org_abs_text_split, actual_abs_text, org_abs_text))


                #break
                    # #dataframe_for_outcomes
                    # x = pd.DataFrame(doc_tag_list, columns=['Labels', 'Outcomes'])
                    # x.index = [os.path.basename(file) for i in range(x.shape[0])]
                    # comet_outcomes_frame = pd.concat([comet_outcomes_frame, x], axis=0)
                    #
                    # y = pd.DataFrame(highlights)
                    # y.index = [os.path.basename(file) for i in range(y.shape[0])]
                    # highlights_frame = pd.concat([highlights_frame, y], axis=0)
                    #
                    # z = pd.DataFrame(tools)
                    # z.index = [os.path.basename(file) for i in range(z.shape[0])]
                    # tools_frame = pd.concat([tools_frame, z], axis=0)

        # comet_outcomes_frame.to_csv(os.path.join(path, 'comet_outcomes.csv'))
        # highlights_frame.to_csv(os.path.join(path, 'highlighted.csv'))
        # tools_frame.to_csv(os.path.join(path, 'tools.csv'))

def case1_case2_case3_preprocess(phrase, tagged_items, starter=None, x=None, outcome_number=None):
    inner_phrase = ''
    phr = phrase.split()
    n = 0
    for v in range(len(phr)):
        curly_braces_text = []
        if v == n:
            if v == 0:
                if not phr[v].startswith('{'):
                    tagged_items.append('B-outcome')
                    inner_phrase += phr[v]
                else:
                    try:
                        if phr[v].startswith('{'):
                            for j in range(v, len(phr)):
                                if not phr[j].endswith('}'):
                                    curly_braces_text.append(phr[j])
                                    tagged_items.append('X')
                                else:
                                    curly_braces_text.append(phr[j])
                                    tagged_items.append('X')
                                    if phr[j+1]:
                                        tagged_items.append('B-outcome')
                                        n = j + 1
                                    else:
                                        n = j
                                    break
                            #print('Curly braces words:', ' '.join(curly_braces_text))
                        else:
                            inner_phrase += ' ' + phr[v]
                            tagged_items.append('I-outcome')
                    except Exception as e:
                        logger.info('Attention {}'.format(e))
            else:
                if phr[v] in get_un_needed():
                    tagged_items.append('Seperator')
                else:
                    try:
                        if phr[v].startswith('{'):
                            for j in range(v, len(phr)):
                                if not phr[j].endswith('}'):
                                    curly_braces_text.append(phr[j])
                                    tagged_items.append('X')
                                else:
                                    curly_braces_text.append(phr[j])
                                    tagged_items.append('X')
                                    n = j
                                    break
                            #print('Curly braces words:', ' '.join(curly_braces_text))
                        else:
                            if outcome_number == -1:
                                if v == len(phr)-1:
                                    tagged_items.append('{}{}-outcome'.format(starter, x))
                                else:
                                    tagged_items.append('I-outcome')
                            else:
                                tagged_items.append('I-outcome')
                            inner_phrase += ' ' + phr[v]
                    except Exception as e:
                        logger.info('Attention {}'.format(e))
            n += 1
    return tagged_items, inner_phrase

def curly_braces(phr, v, inner_phrase, starter, x, curly_braces_text, tagged_items, outcome_number):
    if phr[v].startswith('{'):
        for j in range(v, len(phr)):
            if not phr[j].endswith('}'):
                curly_braces_text.append(phr[j])
                tagged_items.append('X')
            else:
                curly_braces_text.append(phr[j])
                tagged_items.append('X')
                n = j
                break
        # print('Curly braces words:', ' '.join(curly_braces_text))
    else:
        if outcome_number == -1:
            if v == len(phr) - 1:
                tagged_items.append('{}{}-outcome'.format(starter, x))
            else:
                tagged_items.append('I-outcome')
        else:
            tagged_items.append('I-outcome')
        inner_phrase += ' ' + phr[v]

def capture_multi_labels(tag):
    tag_label = None
    if len(tag) == 1:
        tag_label = map_annotations(tag[0])
    elif len(tag) > 1:
        tag_label = map_annotations(tag)
    return tag_label

def de_annotate(text, annotation):
    for x in range(len(annotation)):
        if x in [0, 2]:
            text = re.sub(re.escape(annotation[x]), '', text)
        elif x == 1:
            for i in ['\(E\d+\)', '\(S\d+\)', '\<P\s[\d\,\s]+\>']:
                if re.search(i, annotation[x]):
                    text = re.sub(i, '', text)
    return text

def remove(text, unwanted_patterns):
    for i in unwanted_patterns:
        f = re.compile(i)
        text = f.sub('', text)
    return text

def fetch_highlights(t):
    s = xmltodict.parse(t._p.xml)
    #print(t._p.xml)
    f = s['w:p']
    a, b, c = 'w:r', 'w:rPr', 'w:highlight'
    highlighted_text = []
    for i in (f):
        if i == a:
            if type(f[a]) != list:
                if b in f[a]:
                    if f[a][b] != None:
                        if c in f[a][b]:
                            if type(m['w:t']) == str:
                                #print(m['w:t'])
                                highlighted_text.append(m['w:t'])
            elif type(f[a]) == list:
                for m in f[a]:
                    if b in m:
                        if m[b] != None:
                            if c in m[b]:
                                if type(m['w:t']) == str:
                                    #print(m['w:t'])
                                    highlighted_text.append(m['w:t'])
    return highlighted_text

def get_un_needed():
    needed = ['from', 'of', 'to', 'during']
    stp_punct = [i for i in list(set(stopwords.words('english')))+list(string.punctuation) if i not in needed]
    return stp_punct

def pickup_tools(t):
    tools = []
    for u in re.findall('\[T(.+?)\]', t):
        for q in [r'\<P\s[\d\,\s]+\>', r'\<\/\>', r'\<\\>']:
            r = re.compile(q)
            g = r.findall(u)
            for un in g:
                u = re.sub(un, '', u)
        u = u.strip()
        for q2 in ['^\(E\d+\)', '^\(S\d+\)']:
            u = re.sub(q2, '', u)
        u = re.sub(r'\<\\>', '', u)
        if re.search('\,', u):
            pass
            #print(u)
        tools.append(' '.join(u.split()))
    return tools

def remove_tools_from_outcomes(phrase):
    if re.search('\[T(.+?)\]', phrase):
        o = re.compile('\[T.+?\]')
        phrase = o.sub('', phrase)
    return ' '.join(phrase.split())


def main():
    path = 'comet-data/'
    path_dir = os.path.dirname(path)
    files = os.listdir(path_dir)
    files_list = [path_dir+'/'+i for i in files]
    extract_annotations(files_list, path)

if __name__ == '__main__':
    print(os.getcwd())
    print('Yayixxxxx')
    main()