import gensim


def most_similar(positive=[], negative=[], topn=20):
    w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors_sgd.txt', binary=False)
    for v in w2v.most_similar(positive=positive, negative=negative):
        print(v)

print("Check for queen...")
most_similar(positive=['queen'], topn=10)
print("Check for alice...")
most_similar(positive=['alice'], topn=10)
print("Check for the...")
most_similar(positive=['the'], topn=10)
print("Check for king-he+she...")
most_similar(positive=['king', 'she'], negative=['he'], topn=10)
