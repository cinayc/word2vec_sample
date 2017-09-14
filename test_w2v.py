
import numpy as np
import util_pkg

def most_similar(positive=[], negative=[], topn=20):

    import gensim
    w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors_nadam_corpus.txt', binary=False)
    print('-- by gensim --')
    for v in w2v.most_similar(positive=positive, negative=negative):
        print(v)
    print('----------------')

path = "/home/junsoo/PycharmProjects/word2vec_sample/vectors_nadam_corpus.txt"

util = util_pkg.util()
util.load_model(path=path)
result = util.most_similar(positive=['king', 'woman'], negative=['man'], topn=12, type='cos')

print('--- by mine ---')
for key in result:
    print('%s\t%f' % (key, result[key]))

print('----------------')

most_similar(positive=['king', 'woman'], negative=['man'], topn=10)