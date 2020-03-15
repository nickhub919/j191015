# DOC_BioBERT_DDI
Extracting DDIs from biomedical documents with BioBERT and Multi-Entity Atttntion

BioBERT version = 1.0
python version = 3.5
keras_bert version = 1.4

How to run

1. Download BioBERT model from https://github.com/naver/biobert-pretrained, and copy it to the **BioBERTModel** Folder, note that the corresponding file should be placed in the corresponding folder.
2. Download DDI2013 Corpus and copy it to the **Corpus** Folder,  note that the corresponding file should be placed in the corresponding folder.
4. Use a crawler to download the description documents of the drugs in the corpus from Wikipedia and drugbank, then use Doc2Vec model to change them into vectors, then put the result to the **DrugDocumentEmbedding** folder, we provide the result file *doc_embedding_matrix.rar*.
5. Run *load_data.py* to preprocess the corpus, the processed cropus files are *trainCsentence_token.txt* and *testCsentence_token.txt*. 
6. Run *bert_preprocess_sentence.py* to get the BERT tokens of the corpus, results save as *.pkl*. 
7. Run *main.py* to run the model, the weights of the model will be saved in the **model** folder, the PRF values will be saved in the **res_log** folder. 
