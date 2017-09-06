import numpy as np

class util:

    def __init__(self):
        self.vocab_list = None
        self.vectors = None

    def l2_dist(self, word1_vector, word2_vector):
        return 1 / (1 + np.sqrt(np.sum(np.square(word1_vector - word2_vector))))

    def load_model(self, path):
        with open(path, mode='r') as f:
            vocab_list = list()
            vector_list = list()
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            # print("vocab_size: %d, vector_size: %d" % (vocab_size, vector_size))
            while True:
                line = f.readline()
                line = line.strip()
                if not line: break

                splitted = line.split(" ", vector_size + 1)
                vocab_list.append(splitted[0])
                vector_list.append(splitted[1:])
            print(vocab_list)
            print(np.array(vector_list))

        self.vocab_list =  vocab_list
        self.vectors = np.array(vector_list, dtype=np.float64)

    def make_whole_similarity(self):
        vocab_size = len(self.vocab_list)
        self.sim_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        print(self.sim_matrix)
        print(self.sim_matrix.shape)

        for word1_idx, word1 in enumerate(self.vocab_list):
            word1_vector = self.vectors[word1_idx]

            for word2_idx in range(0, vocab_size):
                word2 = self.vocab_list[word2_idx]
                word2_vector = self.vectors[word2_idx]
                # if word1_idx == 1:
                #     print('%s\t%s' % (word1, word2))
                self.sim_matrix[word1_idx][word2_idx] = self.l2_dist(word1_vector, word2_vector)
                print(self.sim_matrix)

        print(self.sim_matrix)

    def most_similar(self, positive=[], negative=[], topn=10):
        for pos_word in positive:
            pass
        
        for neg_word in negative:
            pass

path = "/home/junsoo/PycharmProjects/word2vec_sample/vectors_adadelta.txt"

util = util()
util.load_model(path=path)
util.make_whole_similarity()


def test1(util):

    word1 = "the"
    word2 = "and"

    word1_vector = util.vectors[util.vocab_list.index(word1)]
    word2_vector = util.vectors[util.vocab_list.index(word2)]

    l2_distance = util.l2_dist(word1_vector, word2_vector)
    print(l2_distance)

# test1(util)