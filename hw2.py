import json
import numpy as np
import sklearn.naive_bayes
from sklearn.base  import TransformerMixin
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import os
list=[]
lableList=[]
lable=['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos',
       'rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian',
       'talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
def solve():
    class_list = np.array(lableList)
    data_list = np.array(list)
    class NLTKBOW(TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return [
                {
                    word: True for word in word_tokenize(document)
                } for document in X
            ]

    pipeline = Pipeline([('布袋模型转换', NLTKBOW()),
                         ('字典列表转矩阵', DictVectorizer(sparse=True)),
                         ('素贝叶斯分类器',sklearn.naive_bayes.MultinomialNB() )],
                        verbose=True
                        )
    scores = cross_val_score(pipeline, data_list, class_list, scoring='accuracy',cv=5)
    print("Score: {}".format(np.mean(scores)))
def getId(string):
    for i in range(len(lable)):
        if(lable[i] in string) :
            return i
def readFile(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    for file in files:  # 遍历文件夹
        file = os.path.join(path, file)

        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = open(file, encoding='utf-8', errors='ignore')  # 打开文件
            iter_f = iter(f)  # 创建迭代器
            str1 = ""
            for line in iter_f:
                str1 = str1 +" "+ line
            list.append(str1)  # 每个文件的文本存到list中
            id=getId(file)
            lableList.append(id)
        else:
            readFile(file)
if __name__ == '__main__':
    path=r'./mini_newsgroups'
    readFile(path)
    solve()