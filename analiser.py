#!/usr/bin/env python
# -*- coding: utf-8 -*-
# sakana baccaに関する投稿分析

import logging
import sys
import os.path
import bz2
import math
import pandas as pd
from gensim import corpora, models
from pprint import pprint

# KH Coderから出力した、文書-単語行列のcsvファイル。
df = pd.read_csv('./data/word-document1007.csv', sep=',')

# 日本語の取り扱い・語の取捨選択はKH Coderでやる方が楽。
# そのため(少々ダサいけれども)文書-単語行列からgensimで扱いやすいDictionaryモデルetcを作成している。
documents = []
words = df.columns.values[1:]  # 一番左のID列は除く
for row in df.values:
    word_counts = row[1:]
    document_bow = []  # 各投稿ごとのbag of words
    for word_index, count in enumerate(word_counts):
        for i in range(count):
            document_bow.append(words[word_index])
    if len(document_bow) > 0:
        documents.append(document_bow)

# 各単語をidに変換する辞書の作成
dictionary = corpora.Dictionary(documents)
pprint(dictionary.token2id)

# documentsをcorpus化する
corpus = list(map(dictionary.doc2bow, documents))

# TF-IDFモデルを作成する。
test_model = models.TfidfModel(corpus)

# corpusへのモデル適用
corpus_tfidf = test_model[corpus]

# id->単語へ変換
texts_tfidf = []  # id -> 単語表示に変えた文書ごとのTF-IDF
for doc in corpus_tfidf:
    text_tfidf = []
    for word in doc:
        text_tfidf.append([dictionary[word[0]], word[1]])
    texts_tfidf.append(text_tfidf)

# 表示
print('===結果表示===')
for text in texts_tfidf:
    print(text)
