from docx import *
import xmltodict
import re
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
import string
import spacy
import os
from tabulate import tabulate


def extract_highlighted_words(files):
    files = [i for i in files if i.endswith('.docx')]
    comet_data = {}
    comet_data_frame = pd.DataFrame()
    outcomes_frame = pd.DataFrame()
    tools_frame = pd.DataFrame()
    for file in files:
        doc = Document(os.path.abspath(file))
        doc_content = doc.paragraphs
        doc_tag_list = []
        outcomes = []
        tools = []
        for s,t in enumerate(doc_content):
            abs_text = t.text
            for v in fetch_highlights(t):
                outcomes.append(v)
            picked_tools = pickup_tools(abs_text)
            if len(picked_tools) > 0:
                for tool in picked_tools:
                    tools.append(tool)
            for ann in re.findall('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\/\>)', abs_text):
                annot = ann[1]
                brace_regex = re.findall('\{.+?\}', annot)
                for _ in brace_regex:
                    f = re.compile(r'\{.+?\}')
                    annot = f.sub('', annot)
                ann = list(ann)
                ann[1] = ' '.join(annot.split())
                annotation = [i.strip() for i in ann]

                if not re.search('\<P\s[\d\,\s]+\>', annotation[1]):
                    tag_no = re.findall(r'\d+', annotation[0])
                    phrase = annotation[1].strip(' ')
                    if len(tag_no) == 1:
                        doc_tag_list.append(('P {}'.format(tag_no[0]), phrase))
                    elif len(tag_no) > 1:
                        for tg in tag_no:
                            doc_tag_list.append(('P {}'.format(tg), phrase))
                else:
                    tags = re.findall('\s?\<P\s[\d\,\s]+\>\s?', annotation[1])
                    tags.insert(0, annotation[0])
                    tags = [i.strip(' <>') for i in tags]
                    if re.search('^\(E\d+\)', annotation[1].strip()):
                        x = int(annotation[1][2])
                        annotation_ = annotation[1][5:]
                        phrases = re.split('\s*\<P\s[\d\,\s]+\>\s*', annotation_)
                        end = phrases[-1].split()
                        last_phrase = [i for i in end if i not in get_un_needed()]
                        for i in range(len(phrases)):
                            if i < len(phrases)-1:
                                inner_phrase = ' '.join([i for i in phrases[i].split() if i not in get_un_needed()] + end[-(x):])
                                # inner_phrase = re.sub('[\{\}\[\]\.]+', '', inner_phrase)
                                #inner_phrase = [i for i in inner_phrase.split() if i not in get_un_needed()]
                                inner_phrase = re.sub('\,','',inner_phrase)
                                doc_tag_list.append((tags[i].strip(), inner_phrase))
                        last_phrase = ' '.join(last_phrase)
                        last_phrase = re.sub('\,', '', last_phrase)
                        doc_tag_list.append((tags[-1], last_phrase))
                    elif re.search('^\(S\d+\)', annotation[1].strip()):
                        x = int(annotation[1][2])
                        annotation_ = annotation[1][5:]
                        phrases = re.split('\s*\<P\s[\d\,\s]+\>\s*', annotation_)
                        start = [i for i in phrases[0].split() if i not in get_un_needed()]
                        for i in range(len(phrases)):
                            if i > 0:
                                inner_phrase = ' '.join(start[:(x)] + [i for i in re.split(' |\,|\.', phrases[i]) if i not in get_un_needed()])
                                # inner_phrase = re.sub('[\{\}\[\]\.]+', '', inner_phrase)
                                # inner_phrase = [i for i in inner_phrase.split() if i not in get_un_needed()]
                                inner_phrase = re.sub('\,', '', inner_phrase)
                                doc_tag_list.append((tags[i], inner_phrase))
                        last_phrase = ' '.join(start)
                        last_phrase = re.sub('\,', '', last_phrase)
                        doc_tag_list.append((tags[0], last_phrase))
                    else:
                        print(annotation, file)
        #dataframe_for_outcomes
        x = pd.DataFrame(doc_tag_list)
        x.index = [os.path.basename(file) for i in range(x.shape[0])]
        comet_data_frame = pd.concat([comet_data_frame, x], axis=0)

        y = pd.DataFrame(outcomes)
        y.index = [os.path.basename(file) for i in range(y.shape[0])]
        outcomes_frame = pd.concat([outcomes_frame, y], axis=0)

        z = pd.DataFrame(tools)
        z.index = [os.path.basename(file) for i in range(z.shape[0])]
        tools_frame = pd.concat([tools_frame, z], axis=0)

    comet_data_frame.to_csv('comet_data.csv')
    outcomes_frame.to_csv('comet_outcomes.csv')
    tools_frame.to_csv('tools.csv')

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
    needed = ['from', 'of']
    stp_punct = [i for i in list(set(stopwords.words('english')))+list(string.punctuation) if i not in needed]
    return stp_punct

def pickup_tools(t):
    tools = []
    for u in re.findall('\[T(.+?)\]', t):
        for q in ['\<P\s[\d\,\s]+\>', '\<\/\>']:
            r = re.compile(q)
            g = r.findall(u)
            for un in g:
                u = re.sub(un, '', u)
        tools.append(' '.join(u.split()))
    return tools

def main():
    path = 'ebm-data/comet-data/'
    path_dir = os.path.dirname(path)
    files = os.listdir(path_dir)
    files_list = [path_dir+'/'+i for i in files]
    extract_highlighted_words(files_list)

if __name__ == '__main__':
    print(os.getcwd())
    #main()