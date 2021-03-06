# ODP-tagger
A tool built for Outcome Detection (a Biomedical named entity recognition task), evaluated for detection of huuman outcomes in clinical trial abstracts. This tagger aims to detect spans of biomedical entities within electronic health/medical text records, written in english language. Such records can be found in Medical Research Abstracts, Clinic letters e.t.c

The tagger adopts a mechanism making use of different variants of Recurrent Neural Networks, (GRU, LSTM). It makes use of biomedically induced feature representations including, 
Word2Vec_pubmed embeddings obtained by training Word2Vec on pubmed abstracts.
Contextual biomedical embeddings obtained by extracting feature representations from the all layers of Biomedical version sof BERT(BioBERT), ELMO(BioELMO) and BioFLAIR. 

## Data
Tested and evaluated on datasets including a health outcomes dataset EBM-COMET.(Outcomes precisely be descirbed as a diagnosis observed prior, during or after a health assessment), EBM-NLP datasethttps://github.com/bepnye/EBM-NLP. all datasets consist of biomedical articles including some describing randomized control trials (RCTs) that compare multiple treatments and a Protein/gene dataset (BC2GM). 


## BioBERT
BioBERT is a biomedical version of BERT built by pre-training a BERT model on biomedical corpora that includes 4.5B words from PubMed abstracts and 13.5B words from PubMed Central (PMC) articles. Pretrained weights of BioBERT model available [here](https://github.com/naver/biobert-pretrained)

## BioELMO
BioELMO is biomedical version of ELMO built by pre-training ELMO on 10M PubMed abstracts ca. 2.64B tokens. ELMO combines both a forward and backward LM to represent a word in a sentence, thereby jointly maximizing the probability of forward and backward directions in a sentence. Pretrained BioELMO configurations [here] (https://github.com/Andy-jqa/bioelmo/blob/master/README.md")
  
## BioFLAIR
BioFLAIR resulted from FLAIR, a character-level bidirectional LM which learns representations (embeddings) of each character within a sequence of words or a sentence. Usage details of this model included [here] (https://github.com/shreyashub/BioFLAIR)
