# BNER-tagger
A tool built for Biomedical named entity recognition. Teseted ab evaluated on datasets including a health outcomes datasets(Outcomes precisely be descirbed as a diagnosis observed prior, during or after a health assessment), Protein/gene dataset (BC2GM) and a dataset .

This tagger aims to detect spans of biomedical entities within electronic health/medical text records, written in english language. Such records can be found in Medical Research Abstracts, Clinic letters e.t.c

The tagger adopts a mechanism making use of different variants of Recurrent Neural Networks, (GRU, LSTM). It makes use of biomedically induced feature representations including, 
Word2Vec_pubmed embeddings, Obtains by training Word2Vec on pubmed abstracts.
Contextual biomedical embeddings obtained by extracting feature representations from the last layer of Biomedical version sof BERT(BioBERT) and ELMO(BioELMO). 

# Data

The dataset could be used for automatic data extraction of the results of a given RCT. This would enable readers to discover the effectiveness of different treatments without needing to read the paper.

The dataset consists of biomedical articles describing randomized control trials (RCTs) that compare multiple treatments. For more information about the dataset, please refer to the original dataset authors github account https://github.com/bepnye/EBM-NLP.

The dataset could be used for automatic data extraction of the results of a given RCT. This would enable readers to discover the effectiveness of different treatments without needing to read the paper.
