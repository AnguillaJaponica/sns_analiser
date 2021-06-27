#import modules
import os.path
from gensim import corpora, models
from gensim.models import LdaModel
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, Rotator
import pprint
import pickle


def create_gensim_lda_model(dictionary, corpus, number_of_topics, words):
    # LDAモデルの作成
    ldamodel = LdaModel(
        corpus,
        num_topics=number_of_topics,
        id2word=dictionary)
    print(ldamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return ldamodel


def compute_coherence_values(
        dictionary,
        corpus,
        documents,
        stop,
        start=1,
        step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def compute_perplexity_values(
        dictionary,
        corpus,
        documents,
        stop,
        start=1,
        step=1):
    perplexity_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary)
        model_list.append(model)
        perplexity_values.append(np.exp2(-model.log_perplexity(corpus)))
    return model_list, perplexity_values


def plot_graph(documents, start, stop, step, dictionary, corpus):
    coherence_model_list, coherence_values = compute_coherence_values(
        dictionary,
        corpus,
        documents,
        stop,
        start,
        step
    )
    perplexity_model_list, perplexity_values = compute_perplexity_values(
        dictionary,
        corpus,
        documents,
        stop,
        start,
        step
    )
    x = range(start, stop, step)
    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, coherence_values, label="Coherence", color="b", marker="^")
    ax2.plot(x, perplexity_values, label="Perplexity", color="g", marker="o")
    ax1.set_xlabel("number of topics")
    ax1.set_ylabel("Coherence")
    ax2.set_ylabel("Perplexity")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.show()


# KH Coderから出力した、文書-単語行列のcsvファイル。
df = pd.read_csv(
    './data/Twitter_gensim.csv',
    sep=',')

# 日本語の取り扱い・語の取捨選択はKH Coderでやる方が楽。
# そのため(少々ダサいけれども)文書-単語行列からgensimで扱いやすいDictionaryモデルを作成している。
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

# documentsをcorpus化する
corpus = list(map(dictionary.doc2bow, documents))

# TF-IDFモデルを作成する。
test_model = models.TfidfModel(corpus)

# corpusへのモデル適用
corpus_tfidf = test_model[corpus]

# 保存
with open("twitter_corpus_tfidf.pkl", 'wb') as f:
    pickle.dump(corpus_tfidf, f)

start, stop, step = 1, 50, 1
plot_graph(documents, start, stop, step, dictionary, corpus_tfidf)

number_of_topics = 7
words = 10
model = create_gensim_lda_model(
    dictionary,
    corpus_tfidf,
    number_of_topics,
    words)
