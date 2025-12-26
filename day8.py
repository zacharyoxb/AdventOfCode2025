""" Day 8 """

from functools import reduce
import networkx as nx
import numpy as np


def day8_part2(distances: list[tuple[float, int, int]], graph: nx.Graph, vectors: np.ndarray):
    """ Part 2 """

    answer = 0

    for dist, i, j in distances:
        graph.add_edge(tuple(vectors[i]), tuple(vectors[j]), weight=dist)
        if nx.number_connected_components(graph) == 1:
            answer = vectors[i][0] * vectors[j][0]
            break
    print(answer)


def day8_part1(distances: list[tuple[float, int, int]], graph: nx.Graph, vectors: np.ndarray):
    """ Part 1 """

    # Add edges for 1000 shortest distances
    for dist, i, j in distances[:1000]:
        graph.add_edge(tuple(vectors[i]), tuple(vectors[j]), weight=dist)

    largest_sizes = []

    # get sizes of 3 largest circuits
    for circuit in nx.connected_components(graph):
        circuit_len = len(circuit)

        # Always try to add to top 3
        if len(largest_sizes) < 3:
            largest_sizes.append(circuit_len)
            largest_sizes.sort()
        elif circuit_len > largest_sizes[0]:
            largest_sizes[0] = circuit_len
            largest_sizes.sort()

    answer = reduce(lambda prev, curr: prev*curr, largest_sizes)

    print(answer)


def day8():
    """ Main function """
    basic_vectors: list[list[int]] = []

    with open("inputs/day8/input.txt", encoding="UTF-8") as f:
        line = f.readline()
        while line != "":
            split_line = line.strip().split(',')
            int_line = list(map(int, split_line))
            basic_vectors.append(int_line)
            line = f.readline()

    vectors = np.array(basic_vectors)

    # make graph
    graph = nx.Graph()
    vec_len = len(vectors)

    distances = []
    for i in range(vec_len):
        for j in range(i+1, vec_len):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            distances.append((dist, i, j))

    distances.sort(key=lambda x: x[0])

    # part 1
    day8_part1(distances, graph, vectors)
    # part 2
    day8_part2(distances, graph, vectors)


if __name__ == "__main__":
    day8()
