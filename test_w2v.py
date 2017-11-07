
import numpy as np
import util_pkg

def most_similar(path, positive=[], negative=[], topn=20):

    import gensim
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    print('-- by gensim --')
    for v in w2v.most_similar(positive=positive, negative=negative):
        print(v)
    print('----------------')

path = "/data/forW2V/vectors_nadam_korean_raw.txt"

util = util_pkg.util()
util.load_model(path=path)
# result = util.most_similar(positive=['서울'], negative=[], topn=12, type='cos')
result = util.most_similar(positive=['한국', '도쿄'], negative=['서울'], topn=12, type='cos')

print('--- by mine ---')
for key in result:
    print('%s\t%f' % (key, result[key]))

print('----------------')

most_similar(path, positive=['한국', '도쿄'], negative=['서울'], topn=10)