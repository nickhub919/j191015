# DOC_BioBERT_DDI
Extracting DDIs from biomedical documents with BioBERT and Multi-Entity Atttntion

How to run

1. Download BioBERT model from https://github.com/naver/biobert-pretrained, and copy it to the **BioBERTModel** Folder, note that the corresponding file should be placed in the corresponding folder.
2. Download DDI2013 Corpus and copy it to the **Corpus** Folder, , note that the corresponding file should be placed in the corresponding folder.
3. Run *load_data.py* to preprocess the corpus, the processed cropus files are *trainCsentence_token.txt* and *testCsentence_token.txt*. 
4. Run *bert_preprocess_sentence.py* to get the BERT tokens of the corpus, results save as *.pkl*. 
5. Run *main.py* to run the model, the weights of the model will be saved in the **model** folder, the PRF values will be saved in the **res_log** folder. 
