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
from pprint import pprint
from tabulate import tabulate


def extract_highlighted_words(files):
    #pick only .docx files and create three seperate data frames outcomes, tools and highlighted words
    files = [i for i in files if i.endswith('.docx')]
    comet_data = {}
    comet_outcomes_frame = pd.DataFrame()
    highlights_frame = pd.DataFrame()
    tools_frame = pd.DataFrame()

    for file in files:
        doc = Document(os.path.abspath(file))
        doc_content = doc.paragraphs #extract content from the word document
        doc_tag_list = []
        highlights = []
        tools = []
        for s,t in enumerate(doc_content):
            abs_text = t.text
            #fetch highlighted words
            for v in fetch_highlights(t):
                highlights.append(v)
            #pick text annotated as tools
            picked_tools = pickup_tools(abs_text)
            if len(picked_tools) > 0:
                for tool in picked_tools:
                    tools.append(tool)
            #look through lines in the paragrpahs, identifying outcomes annotated by <P #>.....</>
            for ann in re.findall('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\/\>)', abs_text):
                annot = ann[1]
                #check nested outcomes outcomes <P #>..<P #>..<\>...</>, detected with closing tag <\>
                if re.search('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\\\>)', annot):
                    tag_no = re.findall(r'\d+', annotation[0])
                    phrase = re.search('(\<P\s[\d\,\s]+\>)(.+?(?=\<))(\<\\\>)', annot)
                    for tg in tag_no:
                        doc_tag_list.append(('P {}'.format(tg), phrase.groups()[1].strip()))
                    annot = re.sub(re.escape(phrase.group().strip()), '', annot)

                #remove content in curly braces {}, because it's not needed in final outcome
                brace_regex = re.findall('\{.+?\}', annot)
                for _ in brace_regex:
                    f = re.compile(r'\{.+?\}')
                    annot = f.sub('', annot)
                ann = list(ann)
                ann[1] = ' '.join(annot.split())
                annotation = [i.strip() for i in ann]

                #dealing with precise outcomes, no additional criteria required to extract outcomes from within openning and closing tags of an outcome
                if not re.search('\<P\s[\d\,\s]+\>', annotation[1]):
                    tag_no = re.findall(r'\d+', annotation[0])
                    phrase = annotation[1].strip(' ')
                    phrase = remove_tools_from_outcomes(phrase)
                    if len(tag_no) == 1:
                        doc_tag_list.append(('P {}'.format(tag_no[0]), phrase))
                    elif len(tag_no) > 1:
                        for tg in tag_no:
                            doc_tag_list.append(('P {}'.format(tg), phrase))

                #dealing with outcomes that either share and ending or a start, additional criteria to process these outcomes
                elif re.search('^\(E\d+\)', annotation[1].strip()) or re.search('^\(S\d+\)', annotation[1].strip()):
                    tags = re.findall('\s?\<P\s[\d\,\s]+\>\s?', annotation[1])
                    tags.insert(0, annotation[0])
                    tags = [i.strip(' <>') for i in tags]
                    x = int(annotation[1][2])
                    annotation_ = annotation[1][5:]

                    if re.search('^\(E\d+\)', annotation[1].strip()):
                        phrases = re.split('\s*\<P\s[\d\,\s]+\>\s*', annotation_)
                        end = phrases[-1].split()
                        last_phrase = [i for i in end if i not in get_un_needed()]
                        for i in range(len(phrases)):
                            if i < len(phrases)-1:
                                inner_phrase = ' '.join([i for i in phrases[i].split() if i not in get_un_needed()] + end[-(x):])
                                inner_phrase = re.sub('\,','',inner_phrase)
                                doc_tag_list.append((tags[i].strip(), inner_phrase))
                        last_phrase = ' '.join(last_phrase)
                        last_phrase = re.sub('\,', '', last_phrase)
                        doc_tag_list.append((tags[-1], last_phrase))

                    elif re.search('^\(S\d+\)', annotation[1].strip()):
                        phrases = re.split('\s*\<P\s[\d\,\s]+\>\s*', annotation_)
                        start = [i for i in phrases[0].split() if i not in get_un_needed()]
                        for i in range(len(phrases)):
                            if i > 0:
                                inner_phrase = ' '.join(start[:(x)] + [i for i in re.split(' |\,|\.', phrases[i]) if i not in get_un_needed()])
                                inner_phrase = re.sub('\,', '', inner_phrase)
                                doc_tag_list.append((tags[i], inner_phrase))
                        last_phrase = ' '.join(start)
                        last_phrase = re.sub('\,', '', last_phrase)
                        doc_tag_list.append((tags[0], last_phrase))
                #any other outcomes
                else:
                    pass
                    print(annotation, file)
        #dataframe_for_outcomes
        x = pd.DataFrame(doc_tag_list, columns=['Labels', 'Outcomes'])
        x.index = [os.path.basename(file) for i in range(x.shape[0])]
        comet_outcomes_frame = pd.concat([comet_outcomes_frame, x], axis=0)

        y = pd.DataFrame(highlights)
        y.index = [os.path.basename(file) for i in range(y.shape[0])]
        highlights_frame = pd.concat([highlights_frame, y], axis=0)

        z = pd.DataFrame(tools)
        z.index = [os.path.basename(file) for i in range(z.shape[0])]
        tools_frame = pd.concat([tools_frame, z], axis=0)

    comet_outcomes_frame.to_csv('comet_outcomes.csv')
    highlights_frame.to_csv('highlighted.csv')
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
            print(u)
        tools.append(' '.join(u.split()))
    return tools

def remove_tools_from_outcomes(phrase):
    if re.search('\[T(.+?)\]', phrase):
        o = re.compile('\[T.+?\]')
        phrase = o.sub('', phrase)
    return ' '.join(phrase.split())


def main():
    path = 'ebm-data/comet-data/'
    path_dir = os.path.dirname(path)
    files = os.listdir(path_dir)
    files_list = [path_dir+'/'+i for i in files]
    extract_highlighted_words(files_list)

if __name__ == '__main__':
    print(os.getcwd())
    main()