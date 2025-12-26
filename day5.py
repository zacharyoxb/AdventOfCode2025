""" Day 5 """


class IntervalTree:
    """ Interval Tree implementation """
    class Node:
        """ Represents node in interval tree """

        def __init__(self, low: int, high: int):
            self.low = low
            self.high = high
            self.max = high
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, low: int, high: int):
        """ Insert an interval into the tree """
        self.root = self._insert(self.root, low, high)

    def _insert(self, node: Node, low: int, high: int):
        if node is None:
            return self.Node(low, high)

        # Check for overlap

        # Use low value as key
        if low < node.low:
            node.left = self._insert(node.left, low, high)
        else:
            node.right = self._insert(node.right, low, high)

        # Update max value
        node.max = max(node.max, high)
        return node

    def contains(self, point: int) -> bool:
        """ Check if point is within any interval """
        return self._contains(self.root, point)

    def _contains(self, node: Node, point: int) -> bool:
        if node is None:
            return False

        # Check if point is in current interval
        if node.low <= point <= node.high:
            return True

        # If left child exists and could contain point
        if node.left and node.left.max >= point:
            return self._contains(node.left, point)

        # Otherwise check right
        return self._contains(node.right, point)

    def get_merged_intervals(self) -> list[tuple[int, int]]:
        """ Get merged intervals directly from tree traversal """
        if self.root is None:
            return []

        intervals = []
        self._collect_merged(self.root, intervals, None)
        return intervals

    def _collect_merged(self, node: Node, intervals: list, last_interval: tuple):
        """ In-order traversal that merges while going through tree """
        if node is None:
            return last_interval

        # Process left subtree first
        last_interval = self._collect_merged(
            node.left, intervals, last_interval)

        # Process current node
        current = (node.low, node.high)

        if last_interval is None:
            # First interval
            intervals.append(current)
            last_interval = current
        else:
            # Check if we can merge with previous
            last_low, last_high = last_interval
            if node.low <= last_high + 1:  # Overlap or adjacent
                # Merge: update the last interval in the list
                merged_low = min(last_low, node.low)
                merged_high = max(last_high, node.high)
                intervals[-1] = (merged_low, merged_high)
                last_interval = (merged_low, merged_high)
            else:
                # New non-overlapping interval
                intervals.append(current)
                last_interval = current

        # Process right subtree
        return self._collect_merged(node.right, intervals, last_interval)


def day5():
    """ Main function """

    tree = IntervalTree()
    fresh_ingredients = 0

    with open("inputs/day5/input.txt", encoding="UTF-8") as f:
        line = f.readline().strip()

        # Insert ranges into interval tree
        while line != "":
            range_start, range_end = line.split('-')
            tree.insert(int(range_start), int(range_end))

            line = f.readline().strip()

        line = f.readline().strip()
        # Test ingredients
        while line != "":
            if tree.contains(int(line)):
                fresh_ingredients += 1
            line = f.readline().strip()

        print(fresh_ingredients)

    # merge all intervals together
    valid_id_tuple = tree.get_merged_intervals()
    valid_ids = 0

    # get amount of valid ids per tuple
    for id_tuple in valid_id_tuple:
        valid_ids += (id_tuple[1] - id_tuple[0])+1

    print(valid_ids)


if __name__ == "__main__":
    day5()
