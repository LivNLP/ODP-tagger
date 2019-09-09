# Health-outcome-tagger
A health outcomes can precisely be descirbed as a diagnosis observed prior, during or after a health assessment.

This tagger aims to detect spans of text pertinent to health outcomes within electronic health/medical text records, written in english language. Such records can be found in Medical Research Abstracts, Clinic letters e.t.c

The tagger adopts an encoder-decoder mechanism making use of different variants of Recurrent Neural Networks, (GRU, LSTM). The encoder uses a couple of resources, glove embeddings pretrained on wikipedia, glove embeddings pretrained on wikipedia and pubmed medical research abstracts, and fasttext pretrained on wikipedia. 

# Data

The dataset could be used for automatic data extraction of the results of a given RCT. This would enable readers to discover the effectiveness of different treatments without needing to read the paper.

The dataset consists of biomedical articles describing randomized control trials (RCTs) that compare multiple treatments. For more information about the dataset, please refer to the original dataset authors github account https://github.com/bepnye/EBM-NLP.

The dataset could be used for automatic data extraction of the results of a given RCT. This would enable readers to discover the effectiveness of different treatments without needing to read the paper.
