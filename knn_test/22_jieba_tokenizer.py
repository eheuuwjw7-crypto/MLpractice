import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import logging

# jieba 内部会打印一些“依赖即将废弃”的提示，这里把它隐藏掉，避免控制台太乱。
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
import jieba

# jieba 第一次分词时会加载词典，这里把普通加载日志关掉，只保留真正需要注意的警告。
jieba.setLogLevel(logging.WARNING)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path


# 项目路径配置：
# __file__ 是当前这个 py 文件的位置。
# parents[1] 表示往上退两级，定位到 ML_Project 目录。
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ORI_DIR = DATA_DIR / "书籍评价.csv"
STOP_DIR = DATA_DIR / "stopwords.txt"


def dm01_train_model():
    # 1. 读取原始评论数据。
    # 这个 csv 文件是 gbk 编码，所以这里要指定 encoding='gbk'，否则中文可能乱码。
    data = pd.read_csv(ORI_DIR,encoding='gbk')
    # print(data)

    # 2. 把文字标签转换成数字标签。
    # 机器学习模型不能直接理解“好评”“差评”这两个字，所以要转成 1 和 0。
    # 好评 -> 1，差评 -> 0
    data['评论标号'] = np.where(data['评价'] == '好评', 1, 0)

    # y 是目标值，也就是模型最终要学习预测的答案。
    y = data['评论标号']
    print(f'{data}')

    # 3. 读取停用词。
    # 停用词就是“的、了、是”这类常见但区分度不高的词。
    # 在做文本分类时，通常会把这些词去掉。
    stopword = []
    with open(STOP_DIR,'r',encoding='utf-8') as f:
        lines = f.readlines()

        # strip() 用来去掉每一行末尾的换行符和空格。
        stopword = [line.strip() for line in lines]

    # set 可以去重，再转回 list，因为 CountVectorizer 的 stop_words 参数需要列表形式。
    stopword = list(set(stopword))

    # 4. 使用 jieba 对中文评论进行分词。
    # 中文句子没有天然空格，比如“这本书很好”。
    # jieba.lcut 会把它切成类似 ['这本书', '很', '好'] 的词列表。
    # ' '.join(...) 再用空格拼起来，变成 CountVectorizer 能识别的文本格式。
    comment_list = [' '.join(jieba.lcut(str(line))) for line in data['内容']]

    # 5. 把分词后的文本转换成数字特征。
    # CountVectorizer 会统计每条评论里每个词出现了多少次。
    # 例如词表里有 ['入门', '小白', '基础']，
    # 某条评论可能被转换成 [1, 1, 0]，表示出现了“入门”和“小白”，没出现“基础”。
    transfer = CountVectorizer(stop_words=stopword)
    x = transfer.fit_transform(comment_list)

    # mynames 是模型最终使用到的词表，也就是每一列数字对应哪个词。
    mynames = transfer.get_feature_names_out()

    # fit_transform 得到的是稀疏矩阵，为了方便观察，这里转成普通数组。
    # 数据量大时不建议随便 toarray，因为会占很多内存；这里数据很小，没问题。
    x = x.toarray()

    # 6. 划分训练集和测试集。
    # 训练集：给模型学习规律。
    # 测试集：模型没见过的数据，用来检查模型效果。
    # test_size=0.2 表示 20% 数据做测试集。
    # random_state=42 表示固定随机种子，保证每次运行切分结果一样。
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    # print(x_train.shape)
    # print(y_train.shape)

    # 7. 创建朴素贝叶斯分类器。
    # MultinomialNB 常用于文本分类，因为文本特征通常是“词出现次数”。
    mymultinomialnb = MultinomialNB()

    # 8. 用训练集训练模型。
    # x_train 是评论转换后的数字特征，y_train 是这些评论对应的真实好评/差评标签。
    mymultinomialnb.fit(x_train,y_train)

    # 9. 用测试集做预测。
    # y_pred 是模型预测出来的标签。
    y_pred = mymultinomialnb.predict(x_test)

    # 10. 输出模型效果。
    # score 对分类问题来说就是准确率：预测正确的数量 / 测试集总数量。
    print(f'准确率：{mymultinomialnb.score(x_test,y_test)}')
    print(f'预测值：{y_pred}')
    print(f'真实值：\n{y_test}')


# 只有直接运行这个文件时，才会执行 dm01_train_model()。
# 如果这个文件被其他文件 import，这里的代码不会自动执行。
if __name__ == '__main__':
    dm01_train_model()
