#~/anaconda2/bin/python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals

import os
import sys
import nltk
import word2vec


### 将docs整理成可以处理的格式 ### doc_id \t doc_content ###
'''
directories = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']

input_file  = open('../data/alldata.txt', 'w')
id_ = 0
for directory in directories:
	rootdir = os.path.join('../data/aclImdb', directory)
	for subdir, dirs, files in os.walk(rootdir):
#		print subdir
#		print dirs
#		print files
		for file_ in files:
			cur_file = os.path.join(subdir, file_)
			if isinstance(cur_file, unicode):
				cur_file = str(cur_file)
			print cur_file, type(cur_file)
			with open(cur_file, 'r') as f:
				doc_id = '_*' + str(id_)
				id_    = id_ + 1
				text   = f.read()
				text   = text.decode('utf-8')
				tokens = nltk.word_tokenize(text) ## 将语句转换为词的list ##
				doc    = ' '.join(tokens).lower()
				doc    = doc.encode('ascii', 'ignore')
				input_file.write(str(doc_id) +' '+ str(doc) +'\n')
input_file.close()
'''

word2vec.doc2vec('../data/alldata.txt', '../data/doc2vec.bin', cbow=0, size=100, window=10, negative=5, hs=0, sample='1e-4', threads=12, iter_=20, min_count=1, verbose=True)


