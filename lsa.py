#import modules
import os.path
from gensim import corpora, models
from gensim.models import LsiModel
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd


def create_gensim_lsa_model(dictionary, corpus, number_of_topics, words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    # generate LSA model
    lsamodel = LsiModel(
        corpus,
        num_topics=number_of_topics,
        id2word=dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel


def compute_coherence_values(
        dictionary,
        corpus,
        documents,
        stop,
        start=2,
        step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def plot_graph(documents, start, stop, step, dictionary, corpus):
    model_list, coherence_values = compute_coherence_values(
        dictionary, corpus, documents, stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


# KH Coderから出力した、文書-単語行列のcsvファイル。
df = pd.read_csv('./data/word-document.csv', sep=',')

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

start, stop, step = 2, 15, 1
# plot_graph(documents, start, stop, step, dictionary, corpus_tfidf)

number_of_topics = 3
words = 5
model = create_gensim_lsa_model(
    dictionary,
    corpus_tfidf,
    number_of_topics,
    words)
