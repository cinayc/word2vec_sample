import codecs

path = '/home/junsoo/PycharmProjects/word2vec_sample/text8'
corpus = codecs.open(path, "r", encoding='utf-8', errors='ignore').read()
words = corpus.split()

with open('text_all_words', 'w') as f:
    for idx,word in enumerate(words):
        idx += 1
        f.write(word+'\n')
