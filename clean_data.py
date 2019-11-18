import pandas as pd
import numpy as np

def sf_age(x) :
    try :
        t = int(x)
    except :
        t = -1
    return t

def sf_gender(x) :
    if x == 'FEMALE' :
        t = 0
    elif x == 'MALE' :
        t = 1
    else :
        t = -1
    return t


def load_and_clean_sub(path):
    labels = pd.read_csv(path, header=None, sep='.txt', encoding='utf-8', engine='python')
    res = pd.DataFrame()
    res['id'] = labels[0]

    tmp = []
    for x in labels[1].values:
        try:
            t = x.split('\t')
            tmp.append([t[1], t[2]])
        except:
            tmp.append([np.nan, np.nan])
    tmp = np.array(tmp)
    res['age'] = tmp[:, 0]
    res['gender'] = tmp[:, 1]

    res['age'] = res['age'].apply(sf_age)
    res['gender'] = res['gender'].apply(sf_gender)
    return res


def load_and_clean_label(path):
    labels = pd.read_csv(path, header=None, sep='.txt', encoding='utf-8', engine='python')
    res = pd.DataFrame()
    res['id'] = labels[0]

    tmp = []
    for x in labels[1].values:
        t = x.split('\t')
        tmp.append([t[1], t[2], t[3:]])
    tmp = np.array(tmp)
    res['age'] = tmp[:, 0]
    res['gender'] = tmp[:, 1]
    res['arrythmia'] = tmp[:, 2]

    res['age'] = res['age'].apply(sf_age)
    res['gender'] = res['gender'].apply(sf_gender)
    return res

#编码标签
#输入：所有类型字典dic，进行编码的标签list
#输出：编码好的标签
def encode(label,dic={}):
    y = np.zeros(( len(label),len(dic.keys()) ))
    list_keys = list(dic.keys())
    for i , y1 in enumerate(label):
        for j in y1:
            index = list_keys.index(j)
            if y[i,index] == 0:
                y[i,index] = y[i,index] + 1
    return y