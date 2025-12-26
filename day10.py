""" Day 10 """
from itertools import chain
import re
from string import ascii_uppercase
import networkx as nx
import numpy as np
from scipy.optimize import linprog

# Helper functions


def get_next_node_add(buttons_pressed: list[str], button_to_add: str) -> str:
    """ Adds button to pressed list and outputs node name """
    buttons_pressed.append(button_to_add)
    buttons_pressed.sort()
    return '^'.join(buttons_pressed)


def get_next_node_remove(buttons_pressed: list[str], button_to_remove: str) -> str:
    """ Removes button from pressed list and outputs node name """
    buttons_pressed.remove(button_to_remove)
    return '^'.join(buttons_pressed)


# Gets a node name based on numbers arg
def get_node_name(buttons: int | tuple[int]) -> str:
    """ Gets name of node listing buttons pressed"""
    ascii_start = 65

    # if int, convert int and return
    if isinstance(buttons, int):
        return chr(ascii_start + (buttons % 26))

    # Otherwise assume list of ints, convert all ints and join
    str_list = [chr(ascii_start + (button % 26)) for button in buttons]
    return '^'.join(str_list)


def visit_leaf_node(graph: nx.DiGraph, node: str, masks: list[int]):
    """ Visits a leaf node, making new node or connecting to previous ones. """
    node_pressed = node.split("^")
    lights_state = graph.nodes[node]["lights_state"]

    for i, mask in enumerate(masks):
        button = ascii_uppercase[i]
        pressed = node_pressed.copy()

        # If the button has already been pressed, and isn't the only one pressed
        if button in pressed and len(pressed) > 1:
            # Remove i from tuple, connect to existing state
            next_node = get_next_node_remove(
                pressed, button)
            graph.add_edge(node, next_node)
        # If the button is already pressed and is only one pressed
        elif button in pressed:
            # connect current node to root
            graph.add_edge(node, "START_STATE")
        # Any other state is new, make new node and connect
        else:
            next_node = get_next_node_add(
                pressed, button)

            # if node is not already there make the node before connecting
            if next_node not in graph:
                next_lights_state = lights_state ^ mask
                graph.add_node(next_node, lights_state=next_lights_state)
            graph.add_edge(node, next_node)


def get_unique_paths(paths: list[str]):
    """ Gets all unique paths from path list """
    # Filter by last node
    unique_paths = []
    seen_last_nodes = set()

    for path in paths:
        last_node = path[-1]
        if last_node not in seen_last_nodes:
            seen_last_nodes.add(last_node)
            unique_paths.append(path)
    return unique_paths


# Machine class

class Machine:
    """ Class representing a machine """
    target_state: int
    buttons: list[list[int]]
    joltages: list[int]
    graph: nx.DiGraph

    def __init__(self, target_state: int, buttons: list[list[int]], joltages: list[int]):
        self.target_state = target_state
        self.buttons = buttons
        self.joltages = joltages
        self.graph = self.construct_machine_graph()

    def construct_machine_graph(self) -> nx.DiGraph:
        """ Constructs a graph representing all the paths to each configuration """
        button_masks = []

        # get amount of buttons in machine
        button_count = max(chain(*self.buttons))

        for button in self.buttons:
            mask = 0
            for digit in button:
                shift = button_count - digit
                mask |= (1 << shift)
            button_masks.append(mask)

        # Build directed graph for all paths
        graph = nx.DiGraph()

        # add 0 state to graph, set visited
        graph.add_node("START_STATE", lights_state=0)

        # Add initial masks named by their buttons
        for i, mask in enumerate(button_masks):
            node_name = get_node_name(i)
            graph.add_node(node_name, lights_state=mask)
            graph.add_edge("START_STATE", node_name)

        # get all unvisited nodes
        leaf_nodes = [node for node in graph.nodes(
        ) if graph.out_degree(node) == 0]

        # visit all currently unvisited nodes
        while len(leaf_nodes) > 0:
            for node in leaf_nodes:
                visit_leaf_node(graph, node, button_masks)

            # get unvisited nodes if still exist
            leaf_nodes = [node for node in graph.nodes(
            ) if graph.out_degree(node) == 0]

        return graph

    def get_fewest_presses(self):
        """ Uses graph to get least amount of buttons needed to press to get to target state """
        try:
            # find nodes with target as lights_state
            target_nodes = [node for node, attrs in self.graph.nodes(
                data=True) if attrs.get('lights_state') == self.target_state]

            # get the closest node
            fewest_presses_node: str = min(target_nodes, key=lambda node: nx.shortest_path_length(
                self.graph, source="START_STATE", target=node))

            # get the amount of buttons pressed from name
            fewest_presses = len(fewest_presses_node.split("^"))

            return fewest_presses
        except nx.NetworkXNoPath:
            return -1

    def get_fewest_joltage_presses(self):
        """ Uses graph to get least amount of buttons needed to press to get to target state
        and adhere to joltage requirements
        """
        try:
            n_counters = len(self.joltages)
            n_buttons = len(self.buttons)

            # Minimise sum of button presses
            c = np.ones(n_buttons)

            # Create coefficient matrix
            a_eq = np.zeros((n_counters, n_buttons), dtype=int)
            for j, button in enumerate(self.buttons):
                for counter in button:
                    a_eq[counter, j] = 1

            result = linprog(c, A_eq=a_eq, b_eq=self.joltages, integrality=1)

            return int(result.fun)
        except nx.NetworkXNoPath:
            return -1


def day10_ext(machines: list[Machine]):
    """ Part 1 """
    total_presses = 0
    total_presses_joltage = 0

    for machine in machines:
        total_presses += machine.get_fewest_presses()
    print(total_presses)

    for machine in machines:
        total_presses_joltage += machine.get_fewest_joltage_presses()
    print(total_presses_joltage)


def day10():
    """ Main function """
    machines_raw: list[str] = []
    machines: list[Machine] = []

    with open("inputs/day10/input.txt", encoding="UTF-8") as f:
        line = f.readline()
        while line != "":
            machines_raw.append(line)
            line = f.readline()

    # convert machines line to machine type
    for line in machines_raw:
        # get target state, convert to binary
        target_state_str = re.search(r'\[([^\]]+)\]', line).group(1)
        target_state = int(target_state_str.replace(
            '.', '0').replace('#', '1'), 2)
        # get buttons
        buttons: list[list[int]] = []
        for match in re.finditer(r'\(([^)]+)\)', line):
            btn_str = match.group(1)
            numbers = [int(x.strip()) for x in btn_str.split(',')]
            buttons.append(numbers)

        # Extract numbers between {}
        joltages_str = re.search(r'\{([^}]+)\}', line).group(1)
        joltages = [int(x.strip()) for x in joltages_str.split(',')]

        machines.append(Machine(target_state, buttons, joltages))

    day10_ext(machines)


if __name__ == "__main__":
    day10()
