# BNER-tagger
A tool built for Biomedical named entity recognition. This tagger aims to detect spans of biomedical entities within electronic health/medical text records, written in english language. Such records can be found in Medical Research Abstracts, Clinic letters e.t.c

The tagger adopts a mechanism making use of different variants of Recurrent Neural Networks, (GRU, LSTM). It makes use of biomedically induced feature representations including, 
Word2Vec_pubmed embeddings, Obtains by training Word2Vec on pubmed abstracts.
Contextual biomedical embeddings obtained by extracting feature representations from the all layers of Biomedical version sof BERT(BioBERT), ELMO(BioELMO) and BioFLAIR. 

#Data
Tested and evaluated on datasets including a health outcomes dataset (COMET-COCHRANE).(Outcomes precisely be descirbed as a diagnosis observed prior, during or after a health assessment), Protein/gene dataset (BC2GM) and an EBM-NLP datasethttps://github.com/bepnye/EBM-NLP. all datasets consist of biomedical articles including some describing randomized control trials (RCTs) that compare multiple treatments. 

