import numpy as np

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams


import codecs
import time
np.random.seed(13)

"""
    for gutenberg data
"""
# path = get_file('alice.txt', origin='http://www.gutenberg.org/files/11/11-0.txt')
# path = '/home/junsoo/PycharmProjects/word2vec_sample/corpus.txt'
# path = '/home/junsoo/PycharmProjects/word2vec_sample/alice.txt'
#
# corpus = codecs.open(path, "r", encoding='utf-8', errors='ignore').readlines()
# corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
# print(corpus)
"""
    for text8 data
"""
path = '/home/junsoo/PycharmProjects/word2vec_sample/text8'
corpus = codecs.open(path, "r", encoding='utf-8', errors='ignore').read()
words = corpus.split()
corpus = []
sentence = ''
for idx,word in enumerate(words):
    idx += 1
    sentence += word + ' '
    if idx % 30 == 0:
        corpus.append(sentence.strip())
        #print(sentence)
        sentence = ''


tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\r\t\n‘“')
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %s' % vocab_size)

dim_embedddings = 128

# inputs
word_inputs = Input(shape=(1,), dtype='int32')
word_vector = Embedding(vocab_size, dim_embedddings, embeddings_initializer='glorot_uniform')(word_inputs)

# context
context_inputs = Input(shape=(1,), dtype='int32')
context_vector = Embedding(vocab_size, dim_embedddings, embeddings_initializer='glorot_uniform')(context_inputs)

output_layer = Dot(axes=2)([word_vector, context_vector])
output_layer = Reshape((1,), input_shape=(1, 1))(output_layer)
output_layer = Activation('sigmoid')(output_layer)

SkipGram = Model(inputs=[word_inputs, context_inputs], outputs=output_layer)
SkipGram.summary()
SkipGram.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

# epochs = 1500   # err: 0.181xxxx, acc: 0.922xxxxxx, optimizer: sgd
# epochs = 20   # err: 0.179222128924, acc: 0.936895229888, optimizer: nadam,   corpus: corpus.txt    [259.537439 sec]
# epochs = 22   # err: 0.183147965964, acc: 0.935174318012, optimizer: nadam,   corpus: corpus.txt    [261.429534 sec]
epochs = 1
t2s = tokenizer.texts_to_sequences(corpus)
len_t2s = len(t2s)

# def generator_for_train(t2s):
#     while 1:
#         for i, doc in enumerate(t2s):
#             data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=4, negative_samples=5.)
#             x = [np.array(x) for x in zip(*data)]
#             y = np.array(labels, dtype=np.int32)
#             yield(x,y)
# """ use fit_generator
#
# """
#
#
# SkipGram.fit_generator(generator=generator_for_train(t2s),
#                        steps_per_epoch=1,
#                        epochs=100,
#                        verbose=1)

""" use train_on_batch
	1/10: 0.247039352477	0.905990441225	[257.527108 sec]
	2/10: 0.234243795963	0.918345201251	[257.891100 sec]
	3/10: 0.224067031795	0.917941807512	[257.579908 sec]
	4/10: 0.216896508488	0.923345649708	[256.295990 sec]
	5/10: 0.214872709133	0.922365269764	[257.950435 sec]
	6/10: 0.2127732193	0.925444639445	[254.762842 sec]
	7/10: 0.20850212332	0.925400217398	[254.441066 sec]
	8/10: 0.206215838865	0.927785103571	[254.892985 sec]
	9/10: 0.200479503969	0.9279399434	[254.332559 sec]
	10/10: 0.19726270359	0.930237522592	[253.945348 sec]
"""
for cur_epoch in range(epochs):
    loss = 0.
    accuracy = 0.

    start_time = time.time()
    for i, doc in enumerate(t2s):
        data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=4, negative_samples=5.)
        x = [np.array(x) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)
        if x:
            train_result = SkipGram.train_on_batch(x, y)
            loss += train_result[0]
            accuracy += train_result[1]

            print("\t%d/%d: %s\t%s" % (i + 1, len_t2s, loss, accuracy))

    avg_loss = loss / len_t2s
    avg_acc = accuracy / len_t2s
    end_time = time.time()
    duration = end_time - start_time

    print("\t%d/%d: %s\t%s\t[%f sec]" % (cur_epoch+1, epochs, avg_loss, avg_acc, duration))

print("Save weights...")
vector_filename = 'vectors_nadam_text8.txt'
f = open(vector_filename ,'w')
f.write('{} {}\n'.format(vocab_size - 1, dim_embedddings))
vectors = SkipGram.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()

