from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from numpy import array
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import gensim

#xml中数据存储相关列表
sentence_id = []
sentence_text = []
entity_id = []
entity_charOffset = []
entity_type = []
entity_text = []
pair_id = []
pair_e1 = []
pair_e2 = []
pair_ddi = []
pair_type=[]



maxlen=154
max_features=20000
EMBEDDING_DIM=200
EMBEDDINGPosition_DIM=50


traindatapath="Train2013/all/"
trainIstanceResultpath=".\\Train2013\\trainIstanceResult.txt"
trainIstancepath=".\\Train2013\\trainIstance.txt"
ProcessedtrainIstancepath=".\\Train2013\\ProcessedtrainIstance.txt"
wordVectorPath=".\wordvector\qian_vec_200.txt"

testIstancepath=".\\Test2013\\testIstance.txt"
ProcessedtestIstancepath=".\\Test2013\\ProcessedtestIstance.txt"
testIstanceResultpath=".\\Test2013\\testIstanceResult.txt"

trainIstanceDrugpath=".\\Train2013\\trainIstance.txt"
testIstanceDrugpath=".\\Test2013\\testIstance.txt"

AllPositionListPath=".\\Train2013\\AllPositionList.txt"
AllPositionMatricxPath=".\\Train2013\\AllPositionMatricx.txt"
AllPosMatricxPath=".\\Train2013\\AllPosMatrix.txt"

trainIstanceDrugPosition1Processed = ".\\Train2013\\ProcessedtrainIstanceDrugPosition1.txt"
trainIstanceDrugPosition2Processed = ".\\Train2013\\ProcessedtrainIstanceDrugPosition2.txt"
testIstanceDrugPosition1Processed = ".\\Test2013\\ProcessedtestIstanceDrugPosition1.txt"
testIstanceDrugPosition2Processed = ".\\Test2013\\ProcessedtestIstanceDrugPosition2.txt"

def loadInstance(fp,fs):
    instance=[]
    instanceResult=[]
    entity1instance=[]
    entity2instance=[]
    result=[]
    with open(fp,'rt',encoding='utf-8') as data_in:
        for line in data_in:
            lines=line.split("$")
            ent1=lines[2].strip((" "))
            ent2=lines[3].strip((" "))
            instanceResult.append(lines[1].strip(" "))
            entity1instance.append(ent1)
            entity2instance.append(ent2)
    data_in.close()
    with open(fs, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            line=line.strip("\n")
            line=line.strip()
            instance.append(line)
    for line in instanceResult:
        instr=line.strip("\n")
        if instr=="none":
            instresult=0
        if instr=="mechanism":
            instresult=1
        if instr=="effect":
            instresult=2
        if instr=="advise":
            instresult=3
        if instr=="int":
            instresult=4
        result.append(instresult)

    #one hot
    result = array(result)
    # one hot encode
    result = to_categorical(result)

    return instance,result,entity1instance,entity2instance

def loaddependecySentence(f1):
    with open(f1, 'rt', encoding='utf-8') as data_in:
        sentence=[]
        for line in data_in:
            sentence.append(line)
    data_in.close()
    return sentence

def loadInstancePosition(fp):   #load pos
    with open(fp,'rt',encoding='utf-8') as data_in:
        position=[]
        for line in data_in:
            line=line.strip("\n")
            position.append(line)
    data_in.close()
    return position

def get_positionweight_input(position):
    positonweight=[]
    for line in position:
        pos = []
        line=line.strip("\n")
        lines=line.split(" ")
        i = 0
        for word in lines:
            i = i+1
            word = float(word)
            pos.append(word)
        while i<154:
            a = float(0)
            i = i+1
            pos.append(a)
        positonweight.append(pos)
    positonweightinput = np.array(positonweight)
    return positonweightinput

def get_position_input(position):
    distance_index = []
    distance=[]
    for i in range(-99, 101):
        distance.append(str(i))
    for i in range(0, 200):
        distance_index.append(str(i))
    word_index=dict(zip(distance,distance_index))
    position_input=[]
    for line in position:
        pos = []
        line=line.strip("\n")
        lines=line.split(" ")
        for word in lines:
            if word in word_index:
                posvalue=word_index.get(word)
                pos.append(posvalue)
        position_input.append(pos)
    position_input=sequence.pad_sequences(position_input,maxlen=maxlen,padding="post")
    return position_input,word_index

def get_pos_input(trainpos,testpos):
    tk = Tokenizer()
    poslist=[]
    with open(".\Train2013\\AllposList.txt", 'rt', encoding='utf-8') as f2:
        for line in f2:
            poslist.append(line)
    f2.close()
    tk.fit_on_texts(poslist)
    train_pos=tk.texts_to_sequences(trainpos)
    test_pos=tk.texts_to_sequences(testpos)
    word_index = tk.word_index
    traininput=sequence.pad_sequences(train_pos,maxlen=maxlen,padding="post")
    testinput=sequence.pad_sequences(test_pos,maxlen=maxlen,padding="post")
    return traininput,testinput,word_index


def get_wordvector_input(TrainInstance,TestInstance):

    tk = Tokenizer(num_words=100000, filters='!"#%&()*+,\'-./:;<=>?@[\]^_`{|}~\t\n', split=" ")
    all=[]
    all.extend(TrainInstance)
    all.extend(TestInstance)
    tk.fit_on_texts(all)
    # file_raw=open('raw.txt','w',encoding='utf-8')
    # for line in all:
    #     raw = text_to_word_sequence(line, filters='!"#%&()*+,\'-./:;<=>?@[\]^_`{|}~\t\n', split=" ")
    #     raw=' '.join(raw)
    #     file_raw.write(raw)
    #     file_raw.write('\n')
    # file_raw.close()
    # tk.fit_on_texts(TrainInstance)
    # tk.fit_on_texts(TestInstance)
    Traininstance = tk.texts_to_sequences(TrainInstance)
    Testinstance = tk.texts_to_sequences(TestInstance)
    word_index = tk.word_index
    print('Pad sequences (samples x time)')
    Traininput = sequence.pad_sequences(Traininstance, maxlen=maxlen,padding="post")
    print('x_train shape:', Traininput.shape)
    print('Pad sequences (samples x time)')
    Testinput = sequence.pad_sequences(Testinstance, maxlen=maxlen,padding="post")
    print('x_test shape:', Testinput.shape)

    return Traininput,Testinput,word_index,tk


def getdependencysentence(tk,trainattensent1,testattensent1):

    trainattensentence1=tk.texts_to_sequences(trainattensent1)
    testattensentence1 = tk.texts_to_sequences(testattensent1)
    trainatteninput1 = sequence.pad_sequences(trainattensentence1, maxlen=28,padding="post")
    testatteninput1 = sequence.pad_sequences(testattensentence1, maxlen=28,padding="post")

    return trainatteninput1,testatteninput1

def get_entityvector_input(entity1train,entity1test,entity2train,entity2test):
    tk = Tokenizer(num_words=100000, filters='!"#$%&()*+,\'-./:;<=>?@[\]^_`{|}~\t\n', split=" ")
    tk.fit_on_texts(entity1train)
    tk.fit_on_texts(entity1test)
    tk.fit_on_texts(entity2train)
    tk.fit_on_texts(entity2test)
    entity1traininstance = tk.texts_to_sequences(entity1train)
    entity1testinstance = tk.texts_to_sequences(entity1test)
    entity2traininstance = tk.texts_to_sequences(entity2train)
    entity2testinstance = tk.texts_to_sequences(entity2test)
    entity_index = tk.word_index
    print('Pad sequences (samples x time)')
    entity1traininput = sequence.pad_sequences(entity1traininstance, maxlen=1,padding="post")
    entity1testinput = sequence.pad_sequences(entity1testinstance, maxlen=1, padding="post")
    entity2traininput = sequence.pad_sequences(entity2traininstance, maxlen=1, padding="post")
    entity2testinput = sequence.pad_sequences(entity2testinstance, maxlen=1, padding="post")
    return entity1traininput,entity1testinput,entity2traininput,entity2testinput,entity_index

def get_embedding_wordvector(word_index):
    embeddings_index = {}
    f = open(wordVectorPath, encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_word = min(max_features, len(word_index))
    embedding_matrix = np.zeros((num_word + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)
    return embedding_matrix,num_word


def produce_matrix(tokenizer):
    all_num=0
    oov=0
    embeddings_index={}
    #myword2vec = gensim.models.KeyedVectors.load_word2vec_format(r'F:\词向量\out_all.bin',unicode_errors='ignore',binary=True)
    myword2vec = gensim.models.KeyedVectors.load_word2vec_format(wordVectorPath,binary=False) #GloVe Model
    #myword2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    print('loaded')
    for i in range(len(myword2vec.index2word)):
        embeddings_index[myword2vec.index2word[i]]=myword2vec.syn0[i]
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 200))
    for word, i in tokenizer.word_index.items():
        all_num+=1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            print(word)
            oov+=1
    print('all is  ',all_num)
    print('the number of not found ',oov)
    return embedding_matrix,all_num



def get_embedding_Posvector(word_index):
    # 创建矩阵
    fM = open(AllPosMatricxPath, encoding='UTF-8')
    fC = open(".\Train2013\\AllposList.txt", encoding='UTF-8')
    embeddings_index = {}
    pos_vector=[]
    pos = []
    for line in fC:
        line = line.strip("\n")
        pos.append(line)
    i = 0
    for line in fM:
        charw = pos[i]
        values = line.split()
        coefs = np.asarray(values[0:], dtype='float32')
        embeddings_index[charw] = coefs
        i = i + 1
    fM.close()
    fC.close()

    # prepare embedding matrix
    num_word = min(max_features, len(word_index))

    embedding_matrix = np.zeros((num_word+1 , EMBEDDINGPosition_DIM))

    for charword, i in word_index.items():
        if i >= max_features:
            continue
        charword=str(charword).upper()
        if charword=="LRB":
            charword="-LRB-"
        if charword=="RRB":
            charword="-RRB-"
        embedding_vector = embeddings_index.get(charword)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    return embedding_matrix,num_word

def get_embedding_Positionvector(word_index):
    # 创建矩阵
    distance_vector=[]
    fC = open(AllPositionMatricxPath, encoding='UTF-8')
    for line in fC:
        line=line.split()
        distance_vector.append(line)
    fC.close()

    # prepare embedding matrix
    num_word = min(max_features, len(word_index))
    embedding_matrix = np.zeros((num_word, EMBEDDINGPosition_DIM))
    for i,j in word_index.items():
        if int(j) >= max_features:
            continue
        embedding_vector = distance_vector[int(j)]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[int(j)] = embedding_vector

    return embedding_matrix,num_word





