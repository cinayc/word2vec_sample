import numpy as np

class util:

    def __init__(self):
        self.vocab_list = None
        self.vectors = None
        self.vector_size = 0
        self.vocab_size = 0

    def l2_dist(self, word1_vector, word2_vector):
        return 1 / (1 + np.sqrt(np.sum(np.square(word1_vector - word2_vector))))

    def cos_similarity(self, word1_vector, word2_vector):
        return (np.dot(word1_vector, word2_vector) / (np.sqrt(np.sum(np.square(word1_vector))) * np.sqrt(np.sum(np.square(word2_vector)))))[0]

    def load_model(self, path):
        with open(path, mode='r') as f:
            vocab_list = list()
            vector_list = list()
            header = f.readline()
            self.vocab_size, self.vector_size = map(int, header.split())
            # print("vocab_size: %d, vector_size: %d" % (self.vocab_size, self.vector_size))
            while True:
                line = f.readline()
                line = line.strip()
                if not line: break

                splitted = line.split(" ", self.vector_size + 1)
                vocab_list.append(splitted[0])
                vector_list.append(splitted[1:])
            # print(vocab_list)
            # print(np.array(vector_list))

        self.vocab_list =  vocab_list
        self.vectors = np.array(vector_list, dtype=np.float64)

    def make_whole_similarity(self, type='l2'):
        self.sim_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        print(self.sim_matrix)
        print(self.sim_matrix.shape)

        for word1_idx, word1 in enumerate(self.vocab_list):
            word1_vector = self.vectors[word1_idx]

            for word2_idx in range(0, self.vocab_size):
                word2 = self.vocab_list[word2_idx]
                word2_vector = self.vectors[word2_idx]
                # if word1_idx == 1:
                #     print('%s\t%s' % (word1, word2))
                if type == 'l2':
                    self.sim_matrix[word1_idx][word2_idx] = self.l2_dist(word1_vector, word2_vector)
                elif type == 'cos':
                    self.sim_matrix[word1_idx][word2_idx] = self.cos_similarity(word1_vector, word2_vector)
                print(self.sim_matrix)

        print(self.sim_matrix)

    def most_similar(self, positive=[], negative=[], topn=10, type='l2'):
        input_vector = np.zeros((1,self.vector_size), dtype=np.float64)
        for pos_word in positive:
            input_vector += self.vectors[self.vocab_list.index(pos_word)]

        for neg_word in negative:
            input_vector -= self.vectors[self.vocab_list.index(neg_word)]

        dist_list = []
        for word_idx, word in enumerate(self.vocab_list):
            word_vector = self.vectors[word_idx]
            if type == 'l2':
                dist = -self.l2_dist(input_vector, word_vector)
            elif type == 'cos':
                dist = -self.cos_similarity(input_vector, word_vector)
            dist_list.append(dist)

        # print(dist_list)
        sorted_dist_idx = np.argsort(dist_list)
        # print(sorted_dist_idx)

        result = {}
        for sorted_idx in sorted_dist_idx[:topn]:
            word = self.vocab_list[sorted_idx]
            dist = dist_list[sorted_idx]
            result[word] = -dist

        return result






if __name__ == '__main__':
    path = "/home/junsoo/PycharmProjects/word2vec_sample/vectors_nadam.txt"

    util = util()
    util.load_model(path=path)
    # util.make_whole_similarity()
    # result = util.most_similar(positive=['king', 'woman'], negative=['man'])
    result = util.most_similar(positive=['queen'], topn=10, type='cos')
    print('----------------')
    print(result)
    print('----------------')

    result = util.most_similar(positive=['alice'], topn=10, type='cos')
    print('----------------')
    print(result)
    print('----------------')

    result = util.most_similar(positive=['king', 'woman'], negative=['man'], topn=10, type='cos')
    print('----------------')
    print(result)
    print('----------------')


    def test1(util):

        word1 = "the"
        word2 = "of"

        word1_vector = util.vectors[util.vocab_list.index(word1)]
        word2_vector = util.vectors[util.vocab_list.index(word2)]

        l2_distance = util.l2_dist(word1_vector, word2_vector)
        print(l2_distance)

    # print('test1')
    # test1(util)

    def test2(util):

        word1 = "the"
        word2 = "of"

        word1_vector = util.vectors[util.vocab_list.index(word1)]
        word2_vector = util.vectors[util.vocab_list.index(word2)]

        distance = util.cos_similarity(word1_vector, word2_vector)
        print(distance)
    # print('test2')
    # test2(util)


    def gensim_most_similar(positive=[], negative=[], topn=20):
        import gensim
        w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        for v in w2v.most_similar(positive=positive, negative=negative):
            print(v)


    # print("Check for queen...")
    # gensim_most_similar(positive=['queen'], topn=10)
    # print("Check for alice...")
    # gensim_most_similar(positive=['alice'], topn=10)
    # print("Check for king-he+she...")
    # gensim_most_similar(positive=['king', 'she'], negative=['he'], topn=10)
