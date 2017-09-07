
import numpy as np
import util_pkg

def most_similar(positive=[], negative=[], topn=20):

    import gensim
    w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors_adadelta.txt', binary=False)
    for v in w2v.most_similar(positive=positive, negative=negative):
        print(v)

"""
print("Check for queen...")
most_similar(positive=['queen'], topn=10)
print("Check for alice...")
most_similar(positive=['alice'], topn=10)
print("Check for rabbit...")
most_similar(positive=['rabbit', 'alice'], negative=['meat'],topn=10)
print("Check for king-man...")
most_similar(positive=['king'], negative=['man'], topn=10)
print("Check for queen-woman...")
most_similar(positive=['queen'], negative=['woman'], topn=10)
print("Check for king-he+she...")
most_similar(positive=['king', 'she'], negative=['he'], topn=10)
print("Check for queen-she+he...")
most_similar(positive=['queen', 'he'], negative=['she'], topn=10)
"""

path = "/home/junsoo/PycharmProjects/word2vec_sample/vectors_adadelta.txt"

util = util_pkg.util()
util.load_model(path=path)
result = util.most_similar(positive=['the'], topn=10)

print('----------------')
print(result)
print('----------------')

most_similar(positive=['the'], topn=10)