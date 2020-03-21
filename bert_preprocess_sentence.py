'''
此文件实现将DDI文本转换成为适合bert训练的文本
1. 生成文本的token id
2. 统计token序列的最大长度233
'''
from keras_bert import Tokenizer
import codecs
import os
import pickle
import math

max_sent_len = 0
org_train_txt = './Train2013/trainCsentence_token.txt'
org_test_txt = './Test2013/testCsentence_token.txt'
save_train_pkl = './Train2013/trainCsentence_bert_token.pkl'
save_test_pkl = './Test2013/testCsentence_bert_token.pkl'

#初始化tokenizer
pretrained_path = './BioBERTModel/pubmed_pmc_470k/'
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)

train_save_list = []
train_segment_save_list = []
train_mask1_list_zhou = []
train_mask2_list_zhou = []
train_mask_save_list = []

test_save_list = []
test_segment_save_list = []
test_mask1_list_zhou = []
test_mask2_list_zhou = []
test_mask_save_list = []

# 修改句子，按照最大长度250来处理，实际最大长度233
with open(org_train_txt, 'rt', encoding='utf-8') as readf:
    for line in readf:
        #newline = line.replace('drug1', '##日')
        #newline = newline.replace('drug2', '##月')
        ids, segments, mask1, mask2 = tokenizer.encode(newline, max_len=250)
        train_save_list.append(ids)
        train_segment_save_list.append(segments)
        train_mask1_list_zhou.append(mask1)
        train_mask2_list_zhou.append(mask2)
        masks = [int(math.ceil(x / (x + 1))) for x in ids]
        train_mask_save_list.append(masks)
with open(save_train_pkl, 'wb') as writef:
    pickle.dump(train_save_list, writef)
    pickle.dump(train_segment_save_list, writef)
    pickle.dump(train_mask_save_list, writef)
    pickle.dump([train_mask1_list_zhou, train_mask2_list_zhou], writef)

with open(org_test_txt, 'rt', encoding='utf-8') as readf:
    for line in readf:
        #newline = line.replace('drug1', '##日')
        #newline = newline.replace('drug2', '##月')
        ids, segments, mask1, mask2 = tokenizer.encode(newline, max_len=250)
        test_save_list.append(ids)
        test_segment_save_list.append(segments)
        test_mask1_list_zhou.append(mask1)
        test_mask2_list_zhou.append(mask2)
        masks = [int(math.ceil(x / (x + 1))) for x in ids]
        test_mask_save_list.append(masks)
with open(save_test_pkl, 'wb') as writef:
    pickle.dump(test_save_list, writef)
    pickle.dump(test_segment_save_list, writef)
    pickle.dump(test_mask_save_list, writef)
    pickle.dump([test_mask1_list_zhou, test_mask2_list_zhou], writef)

print('max sentence len: {0}'.format(max_sent_len))
