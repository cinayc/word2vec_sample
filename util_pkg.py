import numpy as np

class util:

    def __init__(self):
        self.vocab_list = None
        self.vectors = None
        self.vector_size = 0
        self.vocab_size = 0

    def l2_dist(self, word1_vector, word2_vector):
        return 1 / (1 + np.sqrt(np.sum(np.square(word1_vector - word2_vector))))

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

    def make_whole_similarity(self):
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
                self.sim_matrix[word1_idx][word2_idx] = self.l2_dist(word1_vector, word2_vector)
                print(self.sim_matrix)

        print(self.sim_matrix)

    def most_similar(self, positive=[], negative=[], topn=10):
        input_vector = np.zeros((1,self.vector_size), dtype=np.float64)
        for pos_word in positive:
            input_vector += self.vectors[self.vocab_list.index(pos_word)]

        for neg_word in negative:
            input_vector -= self.vectors[self.vocab_list.index(neg_word)]

        l2_dist_list = []
        for word_idx, word in enumerate(self.vocab_list):
            # if word not in positive and word not in negative:
            word_vector = self.vectors[word_idx]
            dist = -self.l2_dist(input_vector, word_vector)
            l2_dist_list.append(dist)

        # print(len(l2_dist_list))
        sorted_l2_dist_idx = np.argsort(l2_dist_list)

        result = {}
        for sorted_idx in sorted_l2_dist_idx[:topn]:
            word = self.vocab_list[sorted_idx]
            l2_dist = l2_dist_list[sorted_idx]
            result[word] = -l2_dist

        return result

if __name__ == '__main__':
    path = "/home/junsoo/PycharmProjects/word2vec_sample/vectors_adadelta.txt"

    util = util()
    util.load_model(path=path)
    # util.make_whole_similarity()
    # result = util.most_similar(positive=['king', 'woman'], negative=['man'])
    result = util.most_similar(positive=['the'], topn=10)

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

    test1(util)