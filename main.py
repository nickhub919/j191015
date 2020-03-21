from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
import os
from keras.optimizers import Adam
from layer.entity_atttion_layer import Entity_Attention
import tensorflow as tf
import numpy as np
from layer.NormalAttention import NormalAttention
from sklearn.model_selection import StratifiedShuffleSplit
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
import keras.backend as K
from keras import Input
from keras.layers import Dense, Embedding, Bidirectional, GRU, concatenate,subtract,multiply,Reshape,Softmax,Dot,Lambda,Permute,Multiply,TimeDistributed,Subtract,Dropout,Add
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
import warnings
from Evalution import getpart_prf, get_prf, get_prf_write
from wordtovector import loadInstance, get_wordvector_input, produce_matrix
from keras_bert import load_trained_model_from_checkpoint

import win_unicode_console
win_unicode_console.enable()
# from keras.activations import relu
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#全局变量
maxlen = 154
max_features = 20000
EMBEDDING_DIM = 200
batch_size = 128

#bert config
layer_num = 12
pretrained_path = './BioBERTModel/pubmed_pmc_470k/'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'biobert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
# wordVectorPath = "./wordvector/public/wikipedia-pubmed-and-PMC-w2v"
trainIstanceDrugpath = "./Train2013/dIstance.txt"
testIstanceDrugpath = "./Test2013/Istance.txt"
trainSentencePath = "./Train2013/trainCsentence_token.txt"
testSentencePath = "./Test2013/testCsentence_token.txt"

#bert tokens
train_pkl = './Train2013/trainCsentence_bert_token.pkl'
test_pkl = './Test2013/testCsentence_bert_token.pkl'
train_entity_pkl = ''
test_entity_pkl = ''


def slice(x, h1, h2):
    return x[:, h1:h2, :]


def dis(e1, e2):
    sub = subtract([e1, e2])
    mut = multiply([e1, e2])
    con = concatenate([sub, mut], axis=-1)
    return con

def self_att_score(input):
    att_socre = Dense(1, activation='tanh')(input)
    print(att_socre)
    att_socre =K.reshape(att_socre,[-1,att_socre.shape[1]])
    print(att_socre)
    att_socre =K.softmax(att_socre,axis=-1)
    print(att_socre)
    return att_socre

def self_att_outshape(inputshape):
    shape1=(list)(inputshape)
    output_shape=[shape1[0],shape1[1]]
    return  tuple(output_shape)


def bulidModel_4():
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    # embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 153, 'h2': 154})(encoded_seq)

    e1_doc_vec= Dense(200,activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(200, activation='relu')(e2_doc_vec)
    distance = dis(e1_doc_vec,e2_doc_vec)
    seq=TimeDistributed(Dense(200,activation='relu'))(encoded_seq)

    att_dot_1=Dot()([seq,e1_doc_vec])
    att_score_1=Softmax()(att_dot_1)
    att_res_1=Dot()(att_score_1,encoded_seq)

    att_dot_2 = Dot()([seq, e2_doc_vec])
    att_score_2 = Softmax()(att_dot_2)
    att_res_2= Dot()(att_score_2, encoded_seq)

    z=concatenate([att_res_1,att_res_2])
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input,e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def bulidModel_3():
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    # embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    e1_doc_vec= Dense(200,activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(200, activation='relu')(e2_doc_vec)

    seq=TimeDistributed(Dense(200,activation='relu'))(encoded_seq)

    att_dot_1=Dot()([seq,e1_doc_vec])
    att_score_1=Softmax()(att_dot_1)
    att_res_1=Dot()(att_score_1,encoded_seq)

    att_dot_2 = Dot()([seq, e2_doc_vec])
    att_score_2 = Softmax()(att_dot_2)
    att_res_2= Dot()(att_score_2, encoded_seq)

    z=concatenate([att_res_1,att_res_2])
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input,e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    myad=Adam()
    print(model.summary())
    return model

def change_shape(input):
    output = K.squeeze(input, axis=1)  # (B, L)
    return output

def out_change_shape(input_shape):
    inputshape=(list)(input_shape)
    out_shape=[inputshape[0],inputshape[2]]
    return tuple(out_shape)


def build_based_model(): #baseline
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    #embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer=Embedding(num_word + 1, 200, mask_zero=True,trainable=False, weights=[embedding_matrix])
    doc_embedding_layer=Embedding(len(doc_vec_embeding),200,mask_zero=True,trainable=True, weights=[doc_vec_embeding])
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 153, 'h2': 154})(encoded_seq)
    slice_1 =Lambda(change_shape,output_shape=out_change_shape)(slice_1)
    z= Dense(256, activation='tanh')(slice_1)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input, e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def gate_mix(input):
    last_step=input[0]
    att_res=input[1]
    att_res=Dense(600,activation='tanh')(att_res)
    rate=Dense(1,activation='sigmoid')(last_step)
    part_1=rate*last_step+(1-rate)*att_res
    return part_1

def gate_outshape(inputshape):
    shape=(list)(inputshape[0])
    outshap=[shape[0],shape[1]]
    return outshap

def build_entity_att(): #71.3
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    #embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    e1_doc_vec = Dense(600,activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(600, activation='relu')(e2_doc_vec)
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 153, 'h2': 154})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    att_e1=NormalAttention()([e1_doc_vec,encoded_seq])
    att_e2 = NormalAttention()([e2_doc_vec, encoded_seq])
    z=concatenate([slice_1,att_e1,att_e2])
    z=Dropout(0.3)(z)
    z = Dense(256 ,activation='tanh')(z)

    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input, e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def build_att_konw_back():
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    # embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    e1_doc_vec = Dense(600, activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(600, activation='relu')(e2_doc_vec)
    sub = Subtract()([e1_doc_vec, e2_doc_vec])
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 153, 'h2': 154})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    all_v=concatenate([sub,e1_doc_vec,e2_doc_vec])
    b=Dense(600,activation='relu')(all_v)

    z = concatenate([slice_1,b])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)

    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input, e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def build_dis_att():
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    # embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    e1_doc_vec = Dense(600, activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(600, activation='relu')(e2_doc_vec)
    sub=Subtract()([e1_doc_vec,e2_doc_vec])
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 153, 'h2': 154})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    att_sub = NormalAttention()([sub, encoded_seq])
    att_e1 = Lambda(my_entity_att, output_shape=out_entity_att)([e1_doc_vec, encoded_seq])
    att_e2 = Lambda(my_entity_att, output_shape=out_entity_att)([e2_doc_vec, encoded_seq])
    z = concatenate([slice_1,att_sub,att_e1,att_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input, e1_kno, e2_kno], outputs=main_output)

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def get_entity_vector_zhou(inputs):
    sequences, entity_mask = inputs
    entity_mask = K.expand_dims(entity_mask, axis=-1)
    return K.sum(sequences * entity_mask, axis=1)

def get_entity_shape(input_shapes):
    B, L, dim = input_shapes[0]
    return tuple((B, dim))


def build_dis_att_with_bert_zhu1():
    '''
    此版本修改了keras-bert底层
    '''
    bert_token_input = Input(shape=(250,), name='bert_token')
    bert_segment_input = Input(shape=(250,), name='bert_segment')
    bert_m1 = Input(shape=[250], name='bert_m1')
    bert_m2 = Input(shape=[250], name='bert_m2')
    bert_model = load_trained_model_from_checkpoint(config_path,
                                                    checkpoint_path,
                                                    seq_len=250)
    for l in bert_model.layers:
        l.trainable = False
    wordVector, embedded = bert_model([bert_token_input, bert_segment_input])

    e1_doc_vec, e2_doc_vec = None, None
    e1_doc_vec = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([embedded, bert_m1])
    e2_doc_vec = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([embedded, bert_m2])
    entity_dense = Dense(768*2, activation='relu')
    e1_doc_vec = entity_dense(e1_doc_vec)
    e2_doc_vec = entity_dense(e2_doc_vec)
    sub=Subtract()([e1_doc_vec,e2_doc_vec])
    # lstm
    encoded_seq = Bidirectional(GRU(768, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 249, 'h2': 250})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    att_sub = NormalAttention()([sub, encoded_seq])
    att_e1 = Lambda(my_entity_att, output_shape=out_entity_att)([e1_doc_vec, encoded_seq])
    att_e2 = Lambda(my_entity_att, output_shape=out_entity_att)([e2_doc_vec, encoded_seq])
    z = concatenate([slice_1,att_sub,att_e1,att_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[bert_token_input, bert_segment_input, bert_m1, bert_m2], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def build_dis_att_with_bert_zhou2():
    '''
    此版本修改了keras-bert底层
    '''
    bert_token_input = Input(shape=(250,), name='bert_token')
    bert_segment_input = Input(shape=(250,), name='bert_segment')
    bert_m1 = Input(shape=[250], name='bert_m1')
    bert_m2 = Input(shape=[250], name='bert_m2')
    bert_model = load_trained_model_from_checkpoint(config_path,
                                                    checkpoint_path,
                                                    seq_len=250)
    for l in bert_model.layers:
        l.trainable = False
    wordVector, embedded = bert_model([bert_token_input, bert_segment_input])

    e1_doc_vec, e2_doc_vec = None, None
    e1_doc_vec = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([embedded, bert_m1])
    e2_doc_vec = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([embedded, bert_m2])
    entity_dense = Dense(768*2, activation='relu')
    e1_doc_vec = entity_dense(e1_doc_vec)
    e2_doc_vec = entity_dense(e2_doc_vec)
    sub=Subtract()([e1_doc_vec,e2_doc_vec])
    # lstm
    encoded_seq = Bidirectional(GRU(768, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 249, 'h2': 250})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    att_sub = NormalAttention()([sub, encoded_seq])
    att_e1 = Lambda(my_entity_att, output_shape=out_entity_att)([e1_doc_vec, encoded_seq])
    att_e2 = Lambda(my_entity_att, output_shape=out_entity_att)([e2_doc_vec, encoded_seq])
    z = concatenate([slice_1,att_sub,att_e1,att_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[bert_token_input, bert_segment_input, bert_m1, bert_m2], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def build_dis_att_with_bert_zhou():

    bert_token_input = Input(shape=(250,), name='bert_token')
    bert_segment_input = Input(shape=(250,), name='bert_segment')
    bert_m1 = Input(shape=[250], name='bert_m1')
    bert_m2 = Input(shape=[250], name='bert_m2')

    bert_model = load_trained_model_from_checkpoint(config_path,
                                                    checkpoint_path,
                                                    seq_len=250)
    for l in bert_model.layers:
        l.trainable = False
    wordVector = bert_model([bert_token_input, bert_segment_input])

    e1_doc_vec, e2_doc_vec = None, None
    e1_doc_vec = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([wordVector, bert_m1])
    e2_doc_vec = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([wordVector, bert_m2])
    entity_dense = Dense(768*2, activation='relu')
    e1_doc_vec = entity_dense(e1_doc_vec)
    e2_doc_vec = entity_dense(e2_doc_vec)

    sub=Subtract()([e1_doc_vec,e2_doc_vec])
    # lstm
    encoded_seq = Bidirectional(GRU(768, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 249, 'h2': 250})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    att_sub = NormalAttention()([sub, encoded_seq])
    att_e1 = Lambda(my_entity_att, output_shape=out_entity_att)([e1_doc_vec, encoded_seq])
    att_e2 = Lambda(my_entity_att, output_shape=out_entity_att)([e2_doc_vec, encoded_seq])
    z = concatenate([slice_1,att_sub,att_e1,att_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[bert_token_input, bert_segment_input, bert_m1, bert_m2], outputs=main_output)

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def build_dis_att_with_bert():
    e1_kno = Input(shape=(1,), dtype='float32', name='kno_e1')
    e2_kno = Input(shape=(1,), dtype='float32', name='kno_e2')
    # main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    bert_token_input = Input(shape=(250,), name='bert_token')
    bert_segment_input = Input(shape=(250,), name='bert_segment')
    # bert_mask_input = Input(shape=(250,), name='bert_mask')
    # embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    # embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    bert_model = load_trained_model_from_checkpoint(config_path,
                                                    checkpoint_path,
                                                    seq_len=250)
    for l in bert_model.layers:
        l.trainable = False
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    e1_doc_vec = Dense(1536, activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(1536, activation='relu')(e2_doc_vec)
    sub=Subtract()([e1_doc_vec,e2_doc_vec])
    # wordVector = embedding_layer(main_input)  # (?,154,200)
    #wordVector = bert_model([bert_token_input, bert_segment_input, bert_mask_input])
    wordVector = bert_model([bert_token_input, bert_segment_input])
    # lstm
    encoded_seq = Bidirectional(GRU(768, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 249, 'h2': 250})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)
    att_sub = NormalAttention()([sub, encoded_seq])
    att_e1 = Lambda(my_entity_att, output_shape=out_entity_att)([e1_doc_vec, encoded_seq])
    att_e2 = Lambda(my_entity_att, output_shape=out_entity_att)([e2_doc_vec, encoded_seq])
    z = concatenate([slice_1,att_sub,att_e1,att_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[bert_token_input, bert_segment_input, e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def build_dis_att_with_bert_and_guo():
    e1_kno = Input(shape=(1,), dtype='float32', name='kno_e1')
    e2_kno = Input(shape=(1,), dtype='float32', name='kno_e2')
    bert_token_input = Input(shape=(250,), name='bert_token')
    bert_segment_input = Input(shape=(250,), name='bert_segment')
    bert_m1 = Input(shape=[250], name='bert_m1')
    bert_m2 = Input(shape=[250], name='bert_m2')
    bert_model = load_trained_model_from_checkpoint(config_path,
                                                    checkpoint_path,
                                                    seq_len=250)
    for l in bert_model.layers:
        l.trainable = False
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    e1_doc_vec = Dense(768, activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(768, activation='relu')(e2_doc_vec)
    # doc_sub = Subtract()([e1_doc_vec, e2_doc_vec])
    e1_bert = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([wordVector, bert_m1])
    e2_bert = Lambda(get_entity_vector_zhou, output_shape=get_entity_shape)([wordVector, bert_m2])
    entity_dense = Dense(768, activation='relu')
    e1_bert_vec = entity_dense(e1_bert)
    e2_bert_vec = entity_dense(e2_bert)
    # bert_sub = Subtract()([e1_bert, e2_bert])
    e1_all = concatenate([e1_doc_vec, e1_bert_vec], axis=-1)
    e2_all = concatenate([e2_doc_vec, e2_bert_vec], axis=-1)
    all_sub = Subtract()([e1_all, e2_all])
    att_all_sub = NormalAttention()([all_sub, encoded_seq])
    att_all_e1 = Lambda(my_entity_att, output_shape=out_entity_att)([e1_all, encoded_seq])
    att_all_e2 = Lambda(my_entity_att, output_shape=out_entity_att)([e2_all, encoded_seq])
    z = concatenate([slice_1, att_all_sub, att_all_e1, att_all_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)
    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[bert_token_input, bert_segment_input, e1_kno, e2_kno, bert_m1, bert_m2], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


# my lambad
def my_entity_att(input):
    entity=input[0]
    seq=input[1]
    entity_rep=K.repeat(entity,250)
    con=concatenate([entity_rep,seq])
    score=Dense(1)(con)
    score = K.squeeze(score, axis=-1)  # (B, L)
    alpha = K.softmax(score, axis=-1)
    Q2 = K.batch_dot(alpha, seq, axes=[1, 1])
    return Q2

def out_entity_att(input_shape):
    entity_shape=input_shape[0]
    outshape=(tuple)(entity_shape)
    return outshape

def build_entity_att_copy():
    e1_kno = Input(shape=(1,), dtype='float32', name='e1_kno')
    e2_kno = Input(shape=(1,), dtype='float32', name='e2_kno')
    main_input = Input(shape=(154,), dtype='float32', name='main_input')  # (?,154)
    #embedding_layer = Embedding(8000 + 1, 200, mask_zero=True, trainable=True)
    embedding_layer = Embedding(num_word + 1, 200, mask_zero=True, trainable=False, weights=[embedding_matrix])
    doc_embedding_layer = Embedding(len(doc_vec_embeding), 200, mask_zero=True, trainable=True,
                                    weights=[doc_vec_embeding])
    e1_doc_vec = doc_embedding_layer(e1_kno)
    e2_doc_vec = doc_embedding_layer(e2_kno)
    e1_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e1_doc_vec)
    e2_doc_vec = Lambda(change_shape, output_shape=out_change_shape)(e2_doc_vec)
    e1_doc_vec = Dense(600, activation='relu')(e1_doc_vec)
    e2_doc_vec = Dense(600, activation='relu')(e2_doc_vec)
    wordVector = embedding_layer(main_input)  # (?,154,200)
    # lstm
    encoded_seq = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(wordVector)
    slice_1 = Lambda(slice, arguments={'h1': 153, 'h2': 154})(encoded_seq)
    slice_1 = Lambda(change_shape, output_shape=out_change_shape)(slice_1)

    att_e1=Lambda(my_entity_att, output_shape=out_entity_att)([e1_doc_vec,encoded_seq])
    att_e2=Lambda(my_entity_att, output_shape=out_entity_att)([e2_doc_vec,encoded_seq])
    z = concatenate([slice_1, att_e1, att_e2])
    z = Dropout(0.3)(z)
    z = Dense(256, activation='tanh')(z)

    main_output = Dense(5, activation='softmax', name='main_output')(z)  # (?,5)
    model = Model(inputs=[main_input, e1_kno, e2_kno], outputs=main_output)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

    # lstm
    z = Bidirectional(GRU(300, dropout=0.3, recurrent_dropout=0.3))(all_vec)  # (?,?,600)
    print("biGRU z:", z.shape)
    distance=dis(e1_kno,e2_kno)
    z=concatenate([z,distance],axis=-1)
    main_output = Dense(5, activation='softmax', name='main_output')(z)   # (?,5)
    print("main_output:", main_output.shape)
    model = Model(inputs=[main_input,e1_pos,e2_pos,e1_kno,e2_kno], outputs=main_output)

    model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def onehotDecoder(predicted):
    predict=[]
    for p in predicted:
        q = p.tolist()
        i = q.index(max(q))
        if i == 0:
            instanceResult = "none"
        if i == 1:
            instanceResult = "mechanism"
        if i == 2:
            instanceResult = "effect"
        if i == 3:
            instanceResult = "advise"
        if i == 4:
            instanceResult = "int"
        predict.append(instanceResult)
    return predict


def trainModel(model, modelName,Traininput, traininstanceResult,
               train_input_e1_os,train_input_e2_pos,
               train_kno_e1,train_kno_e2,
               train_bert_token, train_bert_segment, train_bert_mask, train_bert_zhou_m1, train_bert_zhou_m2,
               train_bert_entity,
               Devinput,DevRes,
               dev_input_e1_pos, dev_input_e2_pos,
               dev_kno_e1,dev_kno_e2,
               dev_bert_token, dev_bert_segment, dev_bert_mask, dev_bert_zhou_m1, dev_bert_zhou_m2,
               dev_bert_entity,
               Testinput,testinstanceResult,
               test_input_e1_pos,test_input_e2_pos,
               test_kno_e1,test_kno_e2,
               test_bert_token, test_bert_segment, test_bert_mask, test_bert_zhou_m1, test_bert_zhou_m2,
               test_bert_entity):
    train_input = {}
    dev_input = {}
    test_input = {}
    train_bert_entity1 = train_bert_entity[:, 0]
    train_bert_entity2 = train_bert_entity[:, 1]
    dev_bert_entity1 = dev_bert_entity[:, 0]
    dev_bert_entity2 = dev_bert_entity[:, 1]
    test_bert_entity1 = test_bert_entity[:, 0]
    test_bert_entity2 = test_bert_entity[:, 1]
    if modelName = 'build_dis_att_with_bert_and_guo':
        train_input = {'bert_token':train_bert_token, 'bert_segment':train_bert_segment,
                       'kno_e1':train_kno_e1, 'kno_e2':train_kno_e2,
                       'bert_m1':train_bert_zhou_m1, 'bert_m2':train_bert_zhou_m2}
        dev_input = {'bert_token':dev_bert_token, 'bert_segment':dev_bert_segment,
                     'kno_e1':dev_kno_e1, 'kno_e2':dev_kno_e2,
                     'bert_m1':dev_bert_zhou_m1, 'bert_m2':dev_bert_zhou_m2}
        test_input = {'bert_token':test_bert_mask, 'bert_segment':test_bert_segment,
                     'kno_e1':test_kno_e1, 'kno_e2':test_kno_e2,
                     'bert_m1':test_bert_zhou_m1, 'bert_m2':test_bert_zhou_m2}

    elif modelName = 'build_dis_att_with_bert_zhou':
        train_input = {'bert_token':train_bert_token, 'bert_segment':train_bert_segment,
                       'bert_m1':train_bert_zhou_m1, 'bert_m2':train_bert_zhou_m2}
        dev_input = {'bert_token':dev_bert_token, 'bert_segment':dev_bert_segment,
                     'bert_m1':dev_bert_zhou_m1, 'bert_m2':dev_bert_zhou_m2}
        test_input = {'bert_token':test_bert_mask, 'bert_segment':test_bert_segment,
                     'bert_m1':test_bert_zhou_m1, 'bert_m2':test_bert_zhou_m2}

    elif modelName = 'build_dis_att_with_bert_zhou2':
        train_input = {'bert_token':train_bert_token, 'bert_segment':train_bert_segment,
                        'bert_m1':train_bert_zhou_m1, 'bert_m2':train_bert_zhou_m2}
        dev_input = {'bert_token':dev_bert_token, 'bert_segment':dev_bert_segment,
                        'bert_m1':dev_bert_zhou_m1, 'bert_m2':dev_bert_zhou_m2}
        test_input = {'bert_token':test_bert_mask, 'bert_segment':test_bert_segment,
                        'bert_m1':test_bert_zhou_m1, 'bert_m2':test_bert_zhou_m2}

    elif modelName = 'build_dis_att_with_bert_zhu1':
        train_input = {'bert_token':train_bert_token, 'bert_segment':train_bert_segment,
                       'bert_m1':train_bert_zhou_m1, 'bert_m2':train_bert_zhou_m2,
                       'bert_entity1':train_bert_entity1, 'bert_entity2':train_bert_entity2}
        dev_input = {'bert_token':dev_bert_token, 'bert_segment':dev_bert_segment,
                     'bert_m1':dev_bert_zhou_m1, 'bert_m2':dev_bert_zhou_m2,
                     'bert_entity1':dev_bert_entity1, 'bert_entity2':dev_bert_entity2}
        test_input = {'bert_token':test_bert_mask, 'bert_segment':test_bert_segment,
                      'bert_m1':test_bert_zhou_m1, 'bert_m2':test_bert_zhou_m2,
                      'bert_entity1':test_bert_entity1, 'bert_entity2':test_bert_entity2}

    elif modelName = 'build_dis_att_with_bert':
        train_input = {'bert_token':train_bert_token, 'bert_segment':train_bert_segment,
                       'kno_e1':train_kno_e1, 'kno_e2':train_kno_e2,}
        dev_input = {'bert_token':dev_bert_token, 'bert_segment':dev_bert_segment,
                     'kno_e1':dev_kno_e1, 'kno_e2':dev_kno_e2}
        test_input = {'bert_token':test_bert_mask, 'bert_segment':test_bert_segment,
                     'kno_e1':test_kno_e1, 'kno_e2':test_kno_e2}
     for i in range(150):
        print("迭代次数：", i+1)
        model.fit(train_input, traininstanceResult, epochs=1, batch_size=batch_size)
        print("dev predict!!")
        devpredict=model.predict(dev_input, verbose=0)
        predict=onehotDecoder(devpredict)
        trueresult=onehotDecoder(DevRes)
        print("get effect prf:")
        getpart_prf("effect", trueresult, predict)
        print("get mechanism prf:")
        getpart_prf("mechanism", trueresult, predict)
        print("get advise prf:")
        getpart_prf("advise", trueresult, predict)
        print("get int prf:")
        getpart_prf("int", trueresult, predict)
        print("get none prf:")
        getpart_prf("none", trueresult, predict)
        print("liu dev prf:")  # 这是预测ddi的prf
        F_dev = get_prf(trueresult, predict)
        print("test predict!!")
        predicted = model.predict({test_input, verbose=0)
        print("write predict!!")
        FNAME = 'predictions-task' + "Epoch" + str(i) + '.txt'
        PREDICTIONSFILE = open(".\predicts\\" + FNAME, "w")
        for p in predicted:
            q=p.tolist()
            j=q.index(max(q))
            if j==0 :
                instanceResult="none"
            if j==1 :
                instanceResult="mechanism"
            if j==2 :
                instanceResult="effect"
            if j==3 :
                instanceResult="advise"
            if j==4 :
                instanceResult="int"
            PREDICTIONSFILE.write("{}\n".format(instanceResult))
        PREDICTIONSFILE.close()

        predict=onehotDecoder(predicted)
        trueresult=onehotDecoder(testinstanceResult)

        print("get effect prf:")
        e_p, e_r, e_f = getpart_prf("effect", trueresult, predict)
        print("get mechanism prf:")
        m_p, m_r, m_f = getpart_prf("mechanism", trueresult, predict)
        print("get advise prf:")
        a_p, a_r, a_f = getpart_prf("advise", trueresult, predict)
        print("get int prf:")
        i_p, i_r, i_f = getpart_prf("int", trueresult, predict)
        print("get none prf:")
        n_p, n_r, n_F = getpart_prf("none", trueresult, predict)
        print("liu test prf:")  # 这是预测ddi的prf
        all_p, all_r, all_f = get_prf(trueresult, predict)
        #为节省磁盘空间，仅存储结果较好的模型
        if all_f > 0.73 :
            F = str(all_f)
            ModelFName = './model/' + modelName + "_epoch-" + str(i) + "F-" + F + ".h5"
            MODELsaveFILEM = open( ModelFName, "w")
            model.save_weights(ModelFName)
            record_path = './res_log/' + modelName + '.txt'
            with open(record_path, 'a') as wf:
                wf.write('effect prf: {0}, {1}, {2}\nmechanism prf: {3}, {4}, {5}\n\
                          advise prf: {6}, {7}, {8}\nint prf: {9}, {10}, {11}\n\
                          none prf: {12}, {13}, {14}\nall prf: {15}, {16}, {17}\n\n'.format(
                          round(e_p, 4), round(e_r, 4), round(e_f, 4),
                          round(m_p, 4), round(m_r, 4), round(m_f, 4),
                          round(a_p, 4), round(a_r, 4), round(a_f, 4),
                          round(i_p, 4), round(i_r, 4), round(i_f, 4),
                          round(n_p, 4), round(n_r, 4), round(n_f, 4),
                          round(all_p, 4), round(all_r, 4), round(all_f, 4)
                          ))


def predict(Input, model_path):
    for i in range(1):
        print("load model !!")
        PREDICTIONSFILEM = open(model_path, "rt")
        model.load_weights(model_path)
        print("predict!!")
        predicted = model.predict(Input, verbose=0)
        print("write predict!!")
        FNAME = 'predictions-task' +  "0.704" + str(i) + '.txt'
        PREDICTIONSFILE = open(".\\loadpredicts\\" + FNAME, "w")
        for p in predicted:
            q=p.tolist()
            i=q.index(max(q))
            if i==0 :
                instanceResult="none"
            if i==1 :
                instanceResult="mechanism"
            if i==2 :
                instanceResult="effect"
            if i==3 :
                instanceResult="advise"
            if i==4 :
                instanceResult="int"
            PREDICTIONSFILE.write("{}\n".format(instanceResult))
        PREDICTIONSFILE.close()
    return predicted


def produce_pos_vector(index,maxlen):
    vector=[]
    start=-index
    for i in range(maxlen):
        vector.append(start)
        start+=1
    return vector

def pos_feature(trainIstanceDrugpath,testIstanceDrugpath):
    maxlen=154
    maxpos=-9999
    minpos=9999
    train_e1_pos_vec=[]
    train_e2_pos_vec=[]
    test_e1_pos_vec=[]
    test_e2_pos_vec = []
    index=0
    for sentence in trainIstanceDrugpath:
        token_list = text_to_word_sequence(sentence,filters='!"#%&()*+,\'-./:;<=>?@[\]^_`{|}~\t\n',lower=True)
        try:
            entity1_pos=token_list.index('drug1')
            maxpos = max(maxpos, entity1_pos)
            minpos = min(minpos, entity1_pos)
            vec=produce_pos_vector(entity1_pos,maxlen)
            train_e1_pos_vec.append(vec)
        except:
            print(sentence)
        try:
            entity2_pos = token_list.index('drug2')
            maxpos = max(maxpos, entity2_pos)
            minpos = min(minpos, entity2_pos)
            vec = produce_pos_vector(entity2_pos,maxlen)
            train_e2_pos_vec.append(vec)
        except:
            print(sentence)

    for sentence in testIstanceDrugpath:
        e1=0
        e2=0
        token_list = text_to_word_sequence(sentence,filters='!"#%&()*+,\'-./:;<=>?@[\]^_`{|}~\t\n',lower=True)
        try:
            entity1_pos=token_list.index('drug1')
            maxpos = max(maxpos, entity1_pos)
            minpos = min(minpos, entity1_pos)
            vec = produce_pos_vector(entity1_pos,maxlen)
            test_e1_pos_vec.append(vec)
            e1+=1
        except:
            print(sentence)
        try:
            entity2_pos = token_list.index('drug2')
            maxpos = max(maxpos, entity2_pos)
            minpos = min(minpos, entity2_pos)
            vec = produce_pos_vector(entity2_pos,maxlen)
            test_e2_pos_vec.append(vec)
            e2+=1
        except:
            print(sentence)
        if(e1!=e2):
            print('a')
    train_e1_pos_vec=np.reshape(train_e1_pos_vec,(len(train_e1_pos_vec),maxlen))
    train_e2_pos_vec=np.reshape(train_e2_pos_vec,(len(train_e2_pos_vec),maxlen))
    test_e1_pos_vec=np.reshape(test_e1_pos_vec,(len(test_e1_pos_vec),maxlen))
    test_e2_pos_vec=np.reshape(test_e2_pos_vec,(len(test_e2_pos_vec),maxlen))
    print('max pos is ',maxpos)
    print('min pos is ',minpos)
    return train_e1_pos_vec,train_e2_pos_vec,test_e1_pos_vec,test_e2_pos_vec

def evalution(predicted,testinstanceResult):
    predict = onehotDecoder(predicted)
    trueresult = onehotDecoder(testinstanceResult)

    print("get effect prf:")
    getpart_prf("effect", trueresult, predict)
    print("get mechanism prf:")
    getpart_prf("mechanism", trueresult, predict)
    print("get advise prf:")
    getpart_prf("advise", trueresult, predict)
    print("get int prf:")
    getpart_prf("int", trueresult, predict)
    print("get none prf:")
    getpart_prf("none", trueresult, predict)
    print("liu prf:")  # 这是预测ddi的prf
    F = get_prf(trueresult, predict)
    return F

def entity_doc_index(entity1,entity2):
    e1_vec=[]
    e2_vec=[]
    file = open('index_dict', 'rb')
    doc_vec_drug=pickle.load(file)
    for i in entity1:
        e1_vec.append(doc_vec_drug[i])
    for i in entity2:
        e2_vec.append(doc_vec_drug[i])
    file.close()
    e1_vec=np.reshape(e1_vec,(len(e1_vec),1))
    e2_vec = np.reshape(e2_vec, (len(e1_vec), 1))
    return e1_vec,e2_vec

if __name__ == "__main__":

# 加载实例
    print("加载实例：")
    traininstance, traininstanceResult,entity1train,entity2train = loadInstance(trainIstanceDrugpath,trainSentencePath)
    print("instance:", traininstance[0])
    print("instanceResult:", traininstanceResult[0])
    print("entity1train:",entity1train[0])
    print("entity2train:",entity2train[0])
    # 加载测试集实例
    testinstance, testinstanceResult,entity1test,entity2test = loadInstance(testIstanceDrugpath,testSentencePath)
    print("instance:", testinstance[0])
    print("instanceResult:", testinstanceResult[0])
    print("entity1test:", entity1test[0])
    print("entity2test:", entity2test[0])

    train_e1_vec,train_e2_vec=entity_doc_index(entity1train,entity2train)
    test_e1_vec,test_e2_vec=entity_doc_index(entity1test,entity2test)

    file = open('./DrugDocumentEmbedding/doc_embedding_matrix', 'rb')
    doc_vec_embeding=pickle.load(file)
    print(len(doc_vec_embeding))

    train_e1_pos_vec,train_e2_pos_vec,test_e1_pos_vec,test_e2_pos_vec=pos_feature(traininstance,testinstance)
    ss=StratifiedShuffleSplit(n_splits=1,test_size=0.1,train_size=0.9)

    # 加载用bert tokenizer处理的token id和token segments
    train_bert_tokens = []
    train_bert_segments = []
    train_bert_masks = []
    test_bert_tokens = []
    test_bert_segments = []
    test_bert_masks = []
    with open(train_pkl, 'rb') as readf:
        train_bert_tokens = pickle.load(readf)
        train_bert_segments = pickle.load(readf)
        train_bert_masks = pickle.load(readf)
        train_bert_zhou_mask = pickle.load(readf)
    with open(test_pkl, 'rb') as readf:
        test_bert_tokens = pickle.load(readf)
        test_bert_segments = pickle.load(readf)
        test_bert_masks = pickle.load(readf)
        test_bert_zhou_mask = pickle.load(readf)
    # 加载用bert tokenizer处理的entity tokens
    train_bert_entities = []
    test_bert_entities = []
    with open(train_entity_pkl, 'rb') as readf:
        train_bert_entities = pickle.load(readf)
    with open(test_entity_pkl, 'rb') as readf:
        test_bert_entities = pickle.load(readf)
    # 获取词输入
    print("获取词输入：")
    Traininput, Testinput, word_index,tk = get_wordvector_input(traininstance,testinstance)
    print("traininput shape:", Traininput.shape)
    print("Traininput[0]:", Traininput[0])
    print("testinput shape:", Testinput.shape)
    print("Testinput[0]:", Testinput[0])

    for train_index, dev_index in ss.split(Traininput, traininstanceResult):
        print("TRAIN:", train_index, "Dev:", dev_index)
        train_kno_e1,dev_kno_e1=np.array(train_e1_vec)[train_index], np.array(train_e1_vec)[dev_index]
        train_kno_e2,dev_kno_e2=np.array(train_e2_vec)[train_index], np.array(train_e2_vec)[dev_index]
        train_input_e1_pos,dev_input_e1_pos=np.array(train_e1_pos_vec)[train_index], np.array(train_e1_pos_vec)[dev_index]
        train_input_e2_pos,dev_input_e2_pos=np.array(train_e2_pos_vec)[train_index], np.array(train_e2_pos_vec)[dev_index]
        train_input_2, dev_input_2 = np.array(Traininput)[train_index], np.array(Traininput)[dev_index]
        train_instacne_res2, dev_instance_res2 = np.array(traininstanceResult)[train_index], np.array(traininstanceResult)[dev_index]

        # 加入对bert输入的切分
        train_bert_tk, dev_bert_tk = np.array(train_bert_tokens)[train_index], np.array(train_bert_tokens)[dev_index]
        train_bert_seg, dev_bert_seg = np.array(train_bert_segments)[train_index], np.array(train_bert_segments)[dev_index]
        train_bert_mk, dev_bert_mk = np.array(train_bert_masks)[train_index], np.array(train_bert_masks)[dev_index]
        train_bert_zhou_m1, dev_bert_zhou_m1 = np.array(train_bert_zhou_mask[0])[train_index], np.array(train_bert_zhou_mask[0])[dev_index]
        train_bert_zhou_m2, dev_bert_zhou_m2 = np.array(train_bert_zhou_mask[1])[train_index], np.array(train_bert_zhou_mask[1])[dev_index]
        train_bert_entities, dev_bert_entities = np.array(train_bert_entities)[train_index], np.array(train_bert_entities)[dev_index]

        print('divide over')
        test_bert_tokens = np.array(test_bert_tokens)
        test_bert_segments = np.array(test_bert_segments)
        test_bert_masks = np.array(test_bert_masks)
        test_bert_zhou_m1 = np.array(test_bert_zhou_mask[0])
        test_bert_zhou_m2 = np.array(test_bert_zhou_mask[1])
        test_bert_entities = np.array(test_bert_entities)

        # 获取词向量embedding input
        # print("获取词向量embedding input:")
        # embedding_matrix, num_word = produce_matrix(tk)
        # print("embedding_matrix shape:",embedding_matrix.shape)

        # build simple model
        model = build_dis_att_with_bert_and_guo()
        trainModel(model, 'build_dis_att_with_bert_and_guo',
                   train_input_2,train_instacne_res2,
                   train_input_e1_pos,train_input_e2_pos,
                   train_kno_e1,train_kno_e2,
                   train_bert_tk, train_bert_seg, train_bert_mk, train_bert_zhou_m1, train_bert_zhou_m2,
                   train_bert_entities,
                   dev_input_2,dev_instance_res2,
                   dev_input_e1_pos,dev_input_e2_pos,
                   dev_kno_e1,dev_kno_e2,
                   dev_bert_tk, dev_bert_seg, dev_bert_mk, dev_bert_zhou_m1, dev_bert_zhou_m2,
                   dev_bert_entities,
                   Testinput,testinstanceResult,
                   test_e1_pos_vec,test_e2_pos_vec,
                   test_e1_vec,test_e2_vec,
                   test_bert_tokens, test_bert_segments, test_bert_masks, test_bert_zhou_m1, test_bert_zhou_m2,
                   test_bert_entities)
        # trainModel(model, Traininput, traininstanceResult, Testinput,testinstanceResult)
        # 根据模型预测
        BERT_Input = {'bert_token':test_bert_tokens,
                      'bert_segment':test_bert_segments,
                      'bert_mask':test_bert_masks,
                      'bert_m1':test_bert_zhou_m1,
                      'bert_m2':test_bert_zhou_m2}
        m_path = './model/...'
        predicted = predict(BERT_Input, m_path)
        # evaltuion
        # for i in range(1):
        evalution(predicted, testinstanceResult)
