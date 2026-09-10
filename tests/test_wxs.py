import numpy as np

from tools.wxs import binary_mutation_distance


def test_binary_mutation_distance_uses_zero_for_two_empty_vectors():
    distance = binary_mutation_distance(np.array([[0, 0, 0], [0, 0, 0]]), 0.0)
    assert distance[0, 1] == 0.0


def test_binary_mutation_distance_keeps_standard_jaccard_for_nonempty_pairs():
    distance = binary_mutation_distance(np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]]), 0.0)
    assert distance[0, 1] == 1.0
    assert distance[0, 2] == 0.5
