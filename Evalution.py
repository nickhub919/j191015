import numpy as np

testIstanceResultpath=".\\Test2013\\protestIstanceResult.txt"
PREDICTIONSFILE = ".\\loadpredicts\\predictions-task0.7370.txt"

def get_prf(true_label=[], predict_label=[]):
    predict_right = 0
    predict_sum = 0
    true_sum = 0
    for index, label in enumerate(true_label):
        if label == "none":
            if predict_label[index] !="none":
                predict_sum += 1
        else:
            true_sum += 1
            if predict_label[index] == label:
                predict_right += 1
                predict_sum += 1
            elif predict_label[index]!="none":
                predict_sum += 1

    if predict_sum == 0 or true_sum == 0:
        print(str(0))
        return 0

    p = float(predict_right) / predict_sum
    r = float(predict_right) / true_sum
    if (p+r) != 0 :
        f = 2 * p * r / (p + r)
    else: f=0
    print("p - " + str(p))
    print("r - " + str(r))
    print("f - " + str(f))
    return p, r, f


def getpart_prf(type, true_label=[], predict_label=[]):
    predict_right = 0
    predict_sum = 0
    true_sum = 0
    for index, label in enumerate(true_label):
        if label == type:
            true_sum += 1
            if predict_label[index] == type:
                predict_right += 1
                predict_sum += 1
        else:
            if predict_label[index] == type:
                predict_sum += 1
    if predict_sum == 0 or true_sum == 0:
        print(str(0))
        return 0
    p = float(predict_right) / predict_sum
    r = float(predict_right) / true_sum
    if (p + r) != 0:
        f = 2 * p * r / (p + r)
    else:
        f = 0
    print("p - " + str(p))
    print("r - " + str(r))
    print("f - " + str(f))
    return p, r, f

def loadResult(fp):
    result=[]
    with open(fp,'rt',encoding='utf-8') as data_in:
        for line in data_in:
            index1 = line.find("[")
            if index1 != -1:
                line = line.replace("[", "")
            index2 = line.find("'")
            if index2 != -1:
                line = line.replace("'", "")
            index3 = line.find("]")
            if index3 != -1:
                line = line.replace("]", "")
            index4 = line.find("\"")
            if index4 != -1:
                line = line.replace("\"", "")
            index4 = line.find("\n")
            if index4 != -1:
                line = line.replace("\n", "")
            result.append(line)
    data_in.close()

    return result

if __name__ == "__main__":

    trueresult=loadResult(testIstanceResultpath)

    print("trueresult",trueresult)
    predict=loadResult(PREDICTIONSFILE)
    print("predict", predict)
    print("get effect prf:")
    getpart_prf("effect",trueresult,predict)
    print("get mechanism prf:")
    getpart_prf("mechanism", trueresult, predict)
    print("get advise prf:")
    getpart_prf("advise", trueresult, predict)
    print("get int prf:")
    getpart_prf("int", trueresult, predict)
    print("get none prf:")
    getpart_prf("none", trueresult, predict)
    print("liu prf:")   #这是预测ddi的prf
    get_prf(trueresult, predict)
