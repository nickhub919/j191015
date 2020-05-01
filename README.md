# JBI-19-1015

**Libraries needed to run this code**

BioBERT version = 1.0  
python version = 3.5  
keras_bert version = 0.71.0  
keras version = 2.1.5  
tensorflow version = 1.12.0  
numpy version = 1.15.4

**Attention**

You need to modify the code of the keras_bert library yourself to change the bert model output to an average pooling of the last four transformer outputs.
1. function:def load_trained_model_from_checkpoint(... **output_layer_num=4**, ...)
2. bert.py function:get_model(): 
...
if len(outputs) > 1:
            #transformed = keras.layers.Concatenate(name='Encoder-Output')(list(reversed(outputs)))
            **transformed = keras.layers.Average(list(reversed(outputs)))**
        else:
            transformed = outputs[0]
        return inputs, transformed
...

**How to run**

1. Download BioBERT model from https://github.com/naver/biobert-pretrained, and copy it to the **BioBERTModel** Folder, note that the corresponding file should be placed in the corresponding folder.
2. Download DDI2013 Corpus and copy it to the **Corpus** Folder,  note that the corresponding file should be placed in the corresponding folder.
4. Use a crawler to download the description documents of the drugs in the corpus from Wikipedia and drugbank, then use Doc2Vec model to change them into vectors, then put the result to the **DrugDocumentEmbedding** folder, we provide our results as an example *doc_embedding_matrix.rar*.
5. Run *load_data.py* to preprocess the corpus, we provide our results as an example *trainCsentence_token.txt* and *testCsentence_token.txt*. 
6. Run *bert_preprocess_sentence.py* to get the BERT tokens of the corpus, results save as *.pkl*. 
7. Run *main.py* to run the model, the weights of the model will be saved in the **model** folder, the PRF values will be saved in the **res_log** folder. 
