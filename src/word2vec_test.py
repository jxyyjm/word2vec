#!/usr/bin/python
# -*- coding:utf-8 -*-

## 试试python的Word2vec模块 ##

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import word2vec
## word2vec.word2phrase -> 将某些词拼成短语 ## 作为包括短语训练预料 ##
#word2vec.word2phrase('../data/text8', '../data/text8-phrases')
## 预测时，model.cosine('los_angeles') ## 会出来类似 san_francisco的词 ##


# 1) 训练 #
## word2vec.word2vec -> 这是训练过程 ## 用包括/not短语的预料库来训练 ##
#word2vec.word2vec('../data/text8-phrases', '../data/text8-phrases.bin', size=100, verbose=True, hs=1, cbow=0)
#word2vec.word2vec('../data/text8', '../data/text8.bin', size=100, verbose=True, hs=1, cbow=0)
## 函数解析 ## help(word2vec.word2vec) ##
#word2vec(train, output, size=100, window=5, sample=u'1e-3', hs=0, negative=5, threads=12, iter_=5, min_count=5, alpha=0.025, debug=2, binary=1, cbow=1, save_vocab=None, read_vocab=None, verbose=False)
'''
	train  -> 训练集 # Use text data from <file> to train the model
	output -> Use <file> to save the resulting word vectors / word clusters
	size   -> 词向量的大小。Set size of word vectors; 
	window -> 上下文的窗口大小。Set max skip length between words;
	sample -> 论文里下采样，随机丢弃高频的词。
	hs     -> 使用分层softmax, Use Hierarchical Softmax,0==not use;
	negative -> 负样本的个数。
	threads  -> 
	min_count-> 出现次数低于多少次，就丢掉该词。
	alpha  -> 学习率
	cbow   -> CBOW，Use the continuous back of words model, 1==skip-gram.
	hs和negative都是用来加快计算的不同方法。
	save_vocab ->
	read_vocab ->
	verbose    ->
'''

# 2) 训练 #
#word2vec.word2clusters('../data/text8', '../data/test8-cluster.txt', classes=10, size=100, verbose=True)

## predictions
model = word2vec.load('../data/text8.bin')
model.vocab    ## 训练预料的向量表示 
model.vectors  ## model的向量表示， model.vectors.shape
model.clusters ## model的那个cluster模型

## 1) model.analogy(pos, neg, n=10)
##	  分析词+-之后的相似词，比如postive = ['king', 'woman']; negative = ['man']
##	  index, metrix = model.analogy(pos, neg, n=10)
##	  model.generate_response(index, metrix).tolist()

## 2) model.cosine(word, n=10) 
##    分析单个词的相似词
##	  index, metrix = model.cosine('dog')
##    model.generate_response(index, metrix).tolist()

## 3) model['dog'] 等同于 model.get_vector('dog')
##	  获取某个词的词向量表示 

## clusters
cluster = word2vec.load_clusters('../data/test8-cluster.txt')
## 1) model.get_cluster(word) 等同于 model['word']
##	  找到某个词所在的类
##    cluster_num = cluster.get_cluster('dog')
## 2) model.get_words_on_cluster(num)
##    找到某个类下的词
##    print cluster_num
#     print cluster.get_words_on_cluster(cluster_num)

