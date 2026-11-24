import csv
from io import StringIO

try:
    from .tree import Node, Tree
except ImportError:  # run as a script: python datafly.py
    from tree import Node, Tree


class _DGH:

    def __init__(self, dgh_path):

        """
        Represents multiple hierarchies as a dictionary of trees.

        :param dgh_path:            Path to the file which contains the DGH definition.
        :raises FileNotFoundError:  If the file is not found.
        :raises IOError:            If the file cannot be read.
        """

        self.hierarchies = dict()
        """
        Dictionary where the values are trees and the keys are the values of the corresponding 
        roots.
        """

        self.gen_levels = dict()
        """
        Dictionary whose keys are the hierarchies root values and whose values are the hierarchies 
        depths (number of generalization levels).
        """

        self._parents = None
        """
        Look up table, built on the first call to generalize(): its keys are couples (generalization
        level, value) and its values are the data of the parent node (None if it's a root).
        """

    def _build_parents(self):

        self._parents = dict()
        for root, tree in self.hierarchies.items():
            stack = [(tree.root, 0)]
            while stack:
                node, depth = stack.pop()
                level = self.gen_levels[root] - depth
                # If a value appears more than once in the same level, the first node found is kept:
                self._parents.setdefault(
                    (level, node.data), None if node.parent is None else node.parent.data)
                stack.extend((child, depth + 1) for child in node.children.values())

    def generalize(self, value, gen_level=None):

        """
        Returns the upper lever generalization of a value in the domain.

        :param value:       Value to generalize.
        :param gen_level:   Current level of generalization, where 0 means it's not generalized.
        :return:            The generalized value on the level above, None if it's a root.
        :raises KeyError:   If the value is not part of the domain.
        """

        if gen_level is not None:
            # Look up table (the same result as searching every tree, but much faster):
            if self._parents is None:
                self._build_parents()
            if (gen_level, value) not in self._parents:
                raise KeyError(value)
            return self._parents[(gen_level, value)]

        # Search across all hierarchies (slow if there are a lot of hierarchies):
        for hierarchy in self.hierarchies:

            # Try to find the node:
            if gen_level is None:
                node = self.hierarchies[hierarchy].bfs_search(value)
            else:
                node = self.hierarchies[hierarchy].bfs_search(
                    value,
                    self.gen_levels[hierarchy] - gen_level)     # Depth.

            if node is None:
                continue
            elif node.parent is None:
                # The value is a hierarchy root:
                return None
            else:
                return node.parent.data

        # The value is not found:
        raise KeyError(value)


class CsvDGH(_DGH):

    def __init__(self, dgh_path):

        super().__init__(dgh_path)

        try:
            with open(dgh_path, 'r') as file:
                for line in file:

                    try:
                        csv_reader = csv.reader(StringIO(line))
                    except IOError:
                        raise
                    values = next(csv_reader)

                    # If it doesn't exist a hierarchy with this root, add one:
                    if values[-1] not in self.hierarchies:
                        self.hierarchies[values[-1]] = Tree(Node(values[-1]))
                        # Add the number of generalization levels:
                        self.gen_levels[values[-1]] = len(values) - 1
                    # Populate hierarchy with the other values:
                    self._insert_hierarchy(values[:-1], self.hierarchies[values[-1]])

        except FileNotFoundError:
            raise
        except IOError:
            raise

    @staticmethod
    def _insert_hierarchy(values, tree):

        """
        Inserts values, ordered from child to parent, to a tree.

        :param values:  List of values to insert.
        :param tree:    Tree where to insert the values.
        :return:        True if the hierarchy has been inserted, False otherwise.
        """

        current_node = tree.root

        for i, value in enumerate(reversed(values)):

            if value in current_node.children:
                current_node = current_node.children[value]
                continue
            else:
                # Insert the hierarchy from this node:
                for v in list(reversed(values))[i:]:
                    current_node.add_child(Node(v))
                    current_node = current_node.children[v]
                return True

        return False
