import numpy as np

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams

import gensim
import codecs
import time
np.random.seed(13)

# path = get_file('alice.txt', origin='http://www.gutenberg.org/files/11/11-0.txt')
path = '/home/junsoo/PycharmProjects/word2vec_sample/corpus.txt'
corpus = codecs.open(path, "r", encoding='utf-8', errors='ignore').readlines()

corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\r\t\n')
tokenizer.fit_on_texts(corpus)
V = len(tokenizer.word_index) + 1
print('Vocabulary size: %s' % V)

dim_embedddings = 128

# inputs
word_inputs = Input(shape=(1,), dtype='int32')
w = Embedding(V, dim_embedddings)(word_inputs)

# context
context_inputs = Input(shape=(1,), dtype='int32')
context  = Embedding(V, dim_embedddings)(context_inputs)
output_layer = Dot(axes=2)([w, context])
output_layer = Reshape((1,), input_shape=(1, 1))(output_layer)
output_layer = Activation('sigmoid')(output_layer)

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
SkipGram = Model(inputs=[word_inputs, context_inputs], outputs=output_layer)
SkipGram.summary()
SkipGram.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

# epochs = 1500   # err: 0.181xxxx, acc: 0.922xxxxxx
epochs = 20
t2s = tokenizer.texts_to_sequences(corpus)
len_t2s = len(t2s)

for cur_epoch in range(epochs):
    loss = 0.
    accuracy = 0.

    start_time = time.time()
    for i, doc in enumerate(t2s):
        data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=4, negative_samples=5.)
        x = [np.array(x) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)
        if x:
            train_result = SkipGram.train_on_batch(x, y)
            loss += train_result[0]
            accuracy += train_result[1]

    avg_loss = loss / len_t2s
    avg_acc = accuracy / len_t2s
    end_time = time.time()
    duration = end_time - start_time

    print("\t%d/%d: %s\t%s\t[%f sec]" % (cur_epoch, epochs, avg_loss, avg_acc, duration))

print("Save weights...")
vector_filename = 'vectors_adadelta.txt'
f = open(vector_filename ,'w')
f.write('{} {}\n'.format(V-1, dim_embedddings))
vectors = SkipGram.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()


# TEST
def most_similar(positive=[], negative=[], topn=20):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(vector_filename, binary=False)
    for v in w2v.most_similar(positive=positive, negative=negative):
        print(v)

print("Check for queen...")
most_similar(positive=['queen'], topn=10)
print("Check for alice...")
most_similar(positive=['alice'], topn=10)
print("Check for king-he+she...")
most_similar(positive=['king', 'she'], negative=['he'], topn=10)


