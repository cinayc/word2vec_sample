import numpy as np

vectors = np.array([[0.130746, 0.309203, 0.457567, 0.645221],
                            [0.0178319, 0.448561, 0.479029, 0.860906],
                            [-0.901462, 0.0169294, 0.440464, 0.0684007],
                            [-0.0240213, 0.454003, 0.207199, 0.498046]])

word_list = ["the", "and", "of", "to"]

def l2_dist(word1_vector, word2_vector):
    return np.sum(np.sqrt(np.square(word1_vector, word2_vector)))

word1 = "the"
word2 = "to"

word1_vector = vectors[word_list.index(word1)]
word2_vector = vectors[word_list.index(word2)]
l2_distance = l2_dist(word1_vector, word2_vector)

print(word1_vector)
print(word2_vector)
print(l2_distance)