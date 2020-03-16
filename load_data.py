from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pathlib import Path
import string

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from lxml import etree
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import xml.etree.ElementTree as ET

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
traindatapath="Train2013/all/"
testdatapath="Test2013/all/"
trainIstanceResultpath=".\\Train2013\\trainIstanceResult.txt"
trainIstancepath=".\\Train2013\\trainIstance.txt"
ProcessedtrainIstancepath=".\\Train2013\\ProcessedtestIstance.txt"


#""" Gather all .xml files in Train folder """
def get_xmls():
    """ Gather all .xml files in Train folder """
    def _get_xmls(path):
        """ Recursive search all .xml files """
        ret = []
        for child in path.iterdir():
            if child.is_dir():
                ret.extend(_get_xmls(child))
            elif child.name.endswith('.xml'):
                ret.append(str(child))
        return ret

    return _get_xmls(Path(traindatapath))

def get_testxmls():
    """ Gather all .xml files in Train folder """
    def _get_xmls(path):
        """ Recursive search all .xml files """
        ret = []
        for child in path.iterdir():
            if child.is_dir():
                ret.extend(_get_xmls(child))
            elif child.name.endswith('.xml'):
                ret.append(str(child))
        return ret

    return _get_xmls(Path(testdatapath))

#Analyze a .xml and return lists about sentence\entity\pair
def analyze_xml(name):
    """
        Analyze a .xml and return lists about sentence\entity\pair
    """
    with open(name, "rb") as fin:
        # parse xml file
        tree = ET.parse(fin)
        root = tree.getroot()
        # print("root:",root.tag, ":", root.attrib)  # 打印根元素的tag和属性document
        # 遍历xml文档的第二层sentence
        for child in root:
            if child.tag=="sentence":
            # for sent in child.iter("sentence"):
                # 第二层节点的标签名称和属性
                sent_text = []
                text=child.get("text")
                sent_text.append(text)
                sentence_text.append(sent_text)
                sent_id = []
                id = child.get("id")
                if id=="DDI-DrugBank.d270.s63" :
                    print(name)
                sent_id.append(id)
                sentence_id.append(sent_id)

                #类成员方法未调试
                # print("sentence:",sentence_text)
                # sentence_text=[]
                # text=sent.get("text")
                # id=sent.get("id")
                # sentence=Sentence(text,id)
                # sentence_text.append(sentence)
                # print("sentence:",sentence_text)

                # 遍历xml文档的第三层entity pair
                for children in child:
                    if children.tag=="entity" :
                        # 第三层节点的标签名称和属性entity
                        ent_text=[]
                        text=children.get("text")
                        ent_text.append(text)
                        entity_text.append(ent_text)
                        ent_id = []
                        ent_id.append(children.get("id"))
                        entity_id.append(ent_id)
                        ent_type = []
                        ent_type.append(children.get("type"))
                        entity_type.append(ent_type)
                        ent_charOffset = []
                        ent_charOffset.append(children.get("charOffset"))
                        entity_charOffset.append(ent_charOffset)
                    # 第三层节点的标签名称和属性pair
                    if children.tag=="pair":
                        pa_id=[]
                        pa_id.append(children.get("id"))
                        pair_id.append(pa_id)
                        pa_e1=[]
                        pa_e1.append(children.get("e1"))
                        pair_e1.append(pa_e1)
                        pa_e2=[]
                        pa_e2.append(children.get("e2"))
                        pair_e2.append(pa_e2)
                        pa_ddi=[]
                        pa_ddi.append(children.get("ddi"))
                        pair_ddi.append(pa_ddi)
                        pa_type=[]
                        if children.get("ddi")=="false":
                            pa_type.append("none")
                        if children.get("ddi")=="true":
                            pa_type.append(children.get("type"))
                        pair_type.append(pa_type)

    return

#compute data number in xml
def range_data(xml,countentity,countpair,countsentence):
# return the number of sentence/entity/pair in xml
    with open(xml, 'rt', encoding='utf-8') as txtFile:
        for line in txtFile:
            if "<entity" in line:
                countentity = countentity + 1
            if "<pair" in line:
                countpair = countpair + 1
            if "<sentence" in line:
                countsentence = countsentence + 1
    return countentity,countpair,countsentence

#load data in xml to lists
def load_data(xmls):

    countentity = 0
    countpair = 0
    countsentence = 0

    for xml in xmls:

        # 解析xml
        analyze_xml(xml)

        #统计xml
        countentity, countpair, countsentence=range_data(xml,countentity, countpair, countsentence)

    print("countentity:", countentity)
    print("countpair:", countpair)
    print("countsentence:", countsentence)

def split_charoffsetAB(charoffset):
    charoffset = charoffset.split(";")
    charoffsetA = charoffset[0]
    charoffsetA1 = charoffsetA.split("-")
    charoffsetA11 = charoffsetA1[0].strip("[\\'")
    charoffsetA12 = charoffsetA1[1]
    charoffsetB = charoffset[1]
    charoffsetB1 = charoffsetB.split("-")
    charoffsetB11 = charoffsetB1[0]
    charoffsetB12 = charoffsetB1[1].strip("\\']")
    return int(charoffsetA11),int(charoffsetA12),int(charoffsetB11),int(charoffsetB12)

def split_charoffset12(charoffset):
    charoffset1 = charoffset.split("-")
    charoffset11 = charoffset1[0]
    charoffset11 = charoffset11.strip("[\\'")
    charoffset12 = charoffset1[1]
    charoffset12 = charoffset12.strip("\\']")
    return int(charoffset11),int(charoffset12)

def change_drug(sentence,charoffset11,charoffset12,drug):
    a=charoffset11
    b=charoffset12
    sentence = sentence[0:a] + drug + sentence[b+1:]
    return sentence

def change_drugs22(sentence,charoffsetA11,charoffsetA12,charoffsetA21, charoffsetA22,charoffsetB11, charoffsetB12,charoffsetB21,charoffsetB22):
    if charoffsetB22>charoffsetB12:
        sentence = change_drug(sentence,charoffsetB21,charoffsetB22,"drug2")
        if charoffsetB12>charoffsetA22:
            sentence = change_drug(sentence, charoffsetB11, charoffsetB12,"drug1")
            if charoffsetA22>charoffsetA12:
                sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
                sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
            else:
                sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
                sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
        else:
            sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
            sentence = change_drug(sentence, charoffsetB11, charoffsetB12,"drug1")
            sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
    else:
        sentence = change_drug(sentence, charoffsetB11, charoffsetB12,"drug1")
        if charoffsetB22>charoffsetA12:
            sentence = change_drug(sentence, charoffsetB21, charoffsetB22,"drug2")
            if charoffsetA22>charoffsetA12:
                sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
                sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
            else:
                sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
                sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
        else:
            sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
            sentence = change_drug(sentence, charoffsetB21, charoffsetB22,"drug2")
            sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
    return sentence

def change_drugs12(sentence,charoffset11,charoffset12,charoffsetA21, charoffsetA22,charoffsetB21,charoffsetB22):
    if charoffsetB22 > charoffset12:
        sentence = change_drug(sentence, charoffsetB21, charoffsetB22,"drug2")
        if charoffsetA22 > charoffset12:
            sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
            sentence = change_drug(sentence, charoffset11, charoffset12,"drug1")
        else:
            sentence = change_drug(sentence, charoffset11, charoffset12,"drug1")
            sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
    else:
        sentence = change_drug(sentence, charoffset11, charoffset12,"drug1")
        sentence = change_drug(sentence, charoffsetB21, charoffsetB22,"drug2")
        sentence = change_drug(sentence, charoffsetA21, charoffsetA22,"drug2")
    return sentence

def change_drugs21(sentence,charoffsetA11,charoffsetA12,charoffsetB11, charoffsetB12,charoffset21,charoffset22):
    if charoffset22 > charoffsetB12:
        sentence = change_drug(sentence, charoffset21, charoffset22,"drug2")
        sentence = change_drug(sentence, charoffsetB11, charoffsetB12,"drug1")
        sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
    else:
        sentence = change_drug(sentence, charoffsetB11, charoffsetB12,"drug1")
        if charoffset22 > charoffsetB12:
            sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
            sentence = change_drug(sentence, charoffset21, charoffset22,"drug2")
        else:
            sentence = change_drug(sentence, charoffset21, charoffset22,"drug2")
            sentence = change_drug(sentence, charoffsetA11, charoffsetA12,"drug1")
    return sentence

def change_drugs11(sentence,charoffset11,charoffset12,charoffset21,charoffset22):
    if charoffset22>charoffset12:
        sentence = change_drug(sentence, charoffset21, charoffset22,"drug2")
        sentence = change_drug(sentence, charoffset11, charoffset12,"drug1")
    else:
        sentence = change_drug(sentence, charoffset11, charoffset12,"drug1")
        sentence = change_drug(sentence, charoffset21, charoffset22,"drug2")
    return sentence

def makeInstanceall():
    #DDI结果
    instanceResult=[]
    instance=[]
    #找目标实体和句子
    for i in range(len(pair_id)):
        if pair_e1[i] != pair_e2[i]:
            #get ddi result
            if pair_ddi=="false":
                result=pair_ddi[i]
            else: result=pair_type[i]
            instanceResult.append(result)

            #get entityid and sentenceid
            e1_id=pair_e1[i]
            e2_id=pair_e2[i]
            pid = str(e1_id)
            sent_id =pid[2:-5]
            sent_id = sent_id.strip(".")


            #get entity through search all list
            for j in range(len(entity_id)):
                if entity_id[j]==e1_id :
                    entity1=entity_text[j]
                if entity_id[j]==e2_id :
                    entity2=entity_text[j]

            #get sentence through search all list
            for k in range(len(sentence_id)):
                sentenceid=str(sentence_id[k])[2:-2]
                #change "drug" in the sentence to "drug1"/"drug2/"drug0"
                if sentenceid==sent_id :
                    sentence=str(sentence_text[k])
                    sentence=sentence.strip("[\\'")
                    sentence=sentence.strip("\\']")

            instancedata = sent_id + "$" +str(result)[2:-2]+"$"+ str(entity1)[2:-2] + "$" + str(entity2)[2:-2] + "$" +sentence
            instance.append(instancedata)

    # 将生成的实例写入文件
    # instancefile=open(".\\Train2013\\alltrainIstance.txt","w")
    # for p in instance:
    #     instancefile.write("{}\n".format(p))
    # instancefile.close()


    # 将生成的实例写入文件
    # instancefile=open(".\\Train2013\\trainIstance.txt","w")
    # for p in instance:
    #     instancefile.write("{}\n".format(p))
    # instancefile.close()
    # instanceResultfile=open(".\\Train2013\\trainIstanceResult.txt","w")
    # for r in instanceResult:
    #     instanceResultfile.write("{}\n".format(r))
    # instanceResultfile.close()

    # 将生成的test实例写入文件
    instancefile=open(".\\Test2013\\testIstanceall.txt","w")
    for p in instance:
        instancefile.write("{}\n".format(p))
    instancefile.close()
    # instanceResultfile=open(".\\Test2013\\testIstanceResult.txt","w")
    # for r in instanceResult:
    #     instanceResultfile.write("{}\n".format(r))
    # instanceResultfile.close()

    return instance,instanceResult



def makeInstance():
    #DDI结果
    instanceResult=[]
    instance=[]
    #找目标实体和句子
    for i in range(len(pair_id)):
        if pair_e1[i] != pair_e2[i]:
            #get ddi result
            if pair_ddi=="false":
                result=pair_ddi[i]
            else: result=pair_type[i]
            instanceResult.append(result)

            #get entityid and sentenceid
            e1_id=pair_e1[i]
            e2_id=pair_e2[i]
            pid = str(e1_id)
            sent_id =pid[2:-5]
            sent_id = sent_id.strip(".")


            #get entity through search all list
            for j in range(len(entity_id)):
                if entity_id[j]==e1_id :
                    entity1=entity_text[j]
                    #find the place which change "drug1"
                    charoffset=str(entity_charOffset[j])
                    index1=charoffset.find(";")
                    if index1!=-1:
                        charoffsetA11, charoffsetA12, charoffsetB11, charoffsetB12 = split_charoffsetAB(charoffset)
                    else :
                        charoffset11, charoffset12=split_charoffset12(charoffset)
                # find the place which change "drug2"
                if entity_id[j]==e2_id :
                    entity2=entity_text[j]
                    # find the place which change "drug2"
                    charoffset = str(entity_charOffset[j])
                    index2 = charoffset.find(";")
                    if index2 != -1:
                        charoffsetA21, charoffsetA22, charoffsetB21, charoffsetB22 = split_charoffsetAB(charoffset)
                    else :
                        charoffset21, charoffset22 = split_charoffset12(charoffset)
            #get sentence through search all list
            for k in range(len(sentence_id)):
                sentenceid=str(sentence_id[k])[2:-2]
                #change "drug" in the sentence to "drug1"/"drug2/"drug0"
                if sentenceid==sent_id :
                    sentence=str(sentence_text[k])
                    sentence=sentence.strip("[\\'")
                    sentence=sentence.strip("\\']")

            if index1!=-1:
                if index2!=-1:
                    sentence=change_drugs22(sentence, charoffsetA11, charoffsetA12, charoffsetA21, charoffsetA22, charoffsetB11,
                               charoffsetB12, charoffsetB21, charoffsetB22)
                else:
                    sentence=change_drugs21(sentence, charoffsetA11, charoffsetA12, charoffsetB11, charoffsetB12, charoffset21,
                                   charoffset22)
            else:
                if index2 != -1:
                    sentence=change_drugs12(sentence, charoffset11, charoffset12, charoffsetA21, charoffsetA22, charoffsetB21,
                                   charoffsetB22)
                else:
                    sentence=change_drugs11(sentence, charoffset11, charoffset12, charoffset21, charoffset22)

            # find the place which change "drug0"
            for j in range(len(entity_id)):
                entity0=str(entity_id[j])
                sent0=sent_id+".e"
                indexE0=entity0.find(sent0)
                if indexE0!=-1 and entity_id[j] != e1_id and entity_id[j] != e2_id:
                    lines = sentence.split(" ")
                    entity0text = entity_text[j]
                    text=str(entity0text)[2:-2]
                    for i in range(len(lines)):
                        word=lines[i].strip(".")
                        word = word.strip(",")
                        word = word.strip(":")
                        word = word.strip("-")
                        word = word.strip("*")
                        if word == text :
                            sen=str(sentence)
                            indexa=sen.find(word)
                            length=indexa+len(word)
                            sentence=sentence[0:indexa]+"drug0"+sentence[length:]

            instancedata = sent_id + "$" +str(result)[2:-2]+"$"+ str(entity1)[2:-2] + "$" + str(entity2)[2:-2] + "$" +sentence
            instance.append(instancedata)

    # # 将生成的实例写入文件
    # instancefile=open(".\\Train2013\\trainIstance.txt","w")
    # for p in instance:
    #     instancefile.write("{}\n".format(p))
    # instancefile.close()


    # 将生成的实例写入文件
    # instancefile=open(".\\Train2013\\trainIstance.txt","w")
    # for p in instance:
    #     instancefile.write("{}\n".format(p))
    # instancefile.close()
    # instanceResultfile=open(".\\Train2013\\trainIstanceResult.txt","w")
    # for r in instanceResult:
    #     instanceResultfile.write("{}\n".format(r))
    # instanceResultfile.close()

    # 将生成的test实例写入文件
    instancefile=open(".\\Test2013\\testIstance.txt","w")
    for p in instance:
        instancefile.write("{}\n".format(p))
    instancefile.close()
    # instanceResultfile=open(".\\Test2013\\testIstanceResult.txt","w")
    # for r in instanceResult:
    #     instanceResultfile.write("{}\n".format(r))
    # instanceResultfile.close()

    return instance,instanceResult

def main():
    """ main routine """
    # 获取所有xml文件路径
   #  print("start to get xml all path")
   #  xmls = get_xmls()
   #  print("xmlsshape:", len(xmls))
   #  print("get xml all path finish!!!")
   # #解析xml文件并加载数据到相应列表中
   #  load_data(xmls)
   #  print("load data finish!!!")

   #  # print("output charoffset:")
   #  # instancefile = open(".\\Train2013\\trainSentence_id.txt", "w")
   #  # for p in sentence_id:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainSentence_text.txt", "w")
   #  # for p in sentence_text:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainEntity_id.txt", "w")
   #  # for p in entity_id:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainEntity_type.txt", "w")
   #  # for p in entity_type:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainEntity_text.txt", "w")
   #  # for p in entity_text:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainPair_id.txt", "w")
   #  # for p in pair_id:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainPair_e1.txt", "w")
   #  # for p in pair_e1:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainPair_e2.txt", "w")
   #  # for p in pair_e2:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainPair_ddi.txt", "w")
   #  # for p in pair_ddi:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #  # instancefile = open(".\\Train2013\\trainPair_type.txt", "w")
   #  # for p in pair_type:
   #  #     instancefile.write("{}\n".format(p))
   #  # instancefile.close()
   #
   #
   #
    #获取实例和结果
    # print("start to make instance")
    # instance, instanceResult=makeInstanceall()
    # print("make instance finish!!!")
   #  #输出验证
   #  print("sentencetext number:", len(sentence_text))
   #  # print("sentencetext:", sentence_text)
   #  print("sentenceid number:", len(sentence_id))
   #  # print("sentence:", sentence_id)
   #  print("entitytext number:", len(entity_text))
   #  # print("entitytext:", entity_text)
   #  print("entityid number:", len(entity_id))
   #  # print("entityid:", entity_id)
   #  print("entitytype number:", len(entity_type))
   #  # print("entitytype:", entity_type)
   #  print("entitycharoffset number:", len(entity_charOffset))
   #  # print("entitycharoffset:", entity_charOffset)
   #  print("pairid number:", len(pair_id))
   #  # print("pair id:", pair_id)
   #  print("paire1 number:", len(pair_e1))
   #  # print("pair e1:", pair_e1)
   #  print("paire2 number:", len(pair_e2))
   #  # print("pair e2:", pair_e2)
   #  print("pairddi number:", len(pair_ddi))
   #  # print("pair ddi:", pair_ddi)
   #  print("pairtype number:", len(pair_type))
   #  # print("pair type:", pair_type)
    #测试集

    testxmls = get_testxmls()
    print("testxmlsshape:", len(testxmls))
    load_data(testxmls)
    # # print("output charoffset:")
    # # instancefile = open(".\\Test2013\\testSentence_id.txt", "w")
    # # for p in sentence_id:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testSentence_text.txt", "w")
    # # for p in sentence_text:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testEntity_id.txt", "w")
    # # for p in entity_id:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testEntity_type.txt", "w")
    # # for p in entity_type:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testEntity_text.txt", "w")
    # # for p in entity_text:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testPair_id.txt", "w")
    # # for p in pair_id:
    # #    instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testPair_e1.txt", "w")
    # # for p in pair_e1:
    # #      instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testPair_e2.txt", "w")
    # # for p in pair_e2:
    # #    instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testPair_ddi.txt", "w")
    # # for p in pair_ddi:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # instancefile = open(".\\Test2013\\testPair_type.txt", "w")
    # # for p in pair_type:
    # #     instancefile.write("{}\n".format(p))
    # # instancefile.close()
    # # 获取实例和结果
    print("start to make instance")
    instance, instanceResult=makeInstanceall()
    print("make instance finish!!!")

    # print("sentencetext number:", len(sentence_text))
    # # print("sentencetext:", sentence_text)
    # print("sentenceid number:", len(sentence_id))
    # # print("sentence:", sentence_id)
    # print("entitytext number:", len(entity_text))
    # # print("entitytext:", entity_text)
    # print("entityid number:", len(entity_id))
    # # print("entityid:", entity_id)
    # print("entitytype number:", len(entity_type))
    # # print("entitytype:", entity_type)
    # print("entitycharoffset number:", len(entity_charOffset))
    # # print("entitycharoffset:", entity_charOffset)
    # print("pairid number:", len(pair_id))
    # # print("pair id:", pair_id)
    # print("paire1 number:", len(pair_e1))
    # # print("pair e1:", pair_e1)
    # print("paire2 number:", len(pair_e2))
    # # print("pair e2:", pair_e2)
    # print("pairddi number:", len(pair_ddi))
    # # print("pair ddi:", pair_ddi)
    # print("pairtype number:", len(pair_type))
    # # print("pair type:", pair_type)

if __name__ == "__main__":
    main()