""" Day 11 """
import re
import networkx as nx


def count_paths_with_both_nodes(graph: nx.DiGraph, start: str, required_nodes: list[str], end: str):
    """
    Count paths from start to end that contain ALL required nodes (any order).
    Graph must be a DAG.
    """
    # Make bitmask to track if required nodes are visited
    n_required = len(required_nodes)
    node_to_bit = {node: 1 << i for i, node in enumerate(required_nodes)}

    topo = list(nx.topological_sort(graph))

    # dp[node][mask] = paths leading to that point
    dp = {node: [0] * (1 << n_required) for node in graph.nodes()}
    dp[start][0] = 1  # Start with no required nodes visited

    # Process in topological order
    for node in topo:
        for mask in range(1 << n_required):
            if dp[node][mask] == 0:
                continue

            for neighbor in graph.successors(node):
                new_mask = mask

                # If neighbor is a required node, mark it visited
                if neighbor in node_to_bit:
                    new_mask |= node_to_bit[neighbor]

                dp[neighbor][new_mask] += dp[node][mask]

    # All bits set = visited all required nodes
    final_mask = (1 << n_required) - 1
    return dp[end][final_mask]


def day11():
    """ Main function """
    graph = nx.DiGraph()

    with open("inputs/day11/input.txt", encoding="UTF-8") as f:
        line = f.readline()
        while line != "":
            values = re.findall(r'[A-Za-z]+', line)
            parent, *children = values
            graph.add_edges_from([(parent, child) for child in children])
            line = f.readline()

    # part 1
    paths = list(nx.all_simple_paths(graph, "you", "out"))
    total_paths = len(paths)
    print(total_paths)

    # part 2
    total_valid_paths = count_paths_with_both_nodes(
        graph, "svr", ["dac", "fft"], "out")
    print(total_valid_paths)


if __name__ == "__main__":
    day11()
