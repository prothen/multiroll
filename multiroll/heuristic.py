#!/usr/bin/env python
"""
    Heuristic with networkX graph shortest_path algorithm.

    Author: Philipp Rothenhäusler, Stockholm 2020
"""


import networkx


from multiroll import *

from .constants import *
from .framework import *
from .coordinate import *
from .agent import *
from .edge import *


class ShortestPath(multiroll.graph.Graph):
    """Shortest path algorithm based on networX.

        Todo:
             Use all_shortest_path computation

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _path_from_networkx(self, path):
        """ Convert the networkX tuples into edge_container references. """

        path_nodes = list()
        heuristic = dict()
        for idx, node in enumerate(path):
            if idx == len(path)-1:
                continue
            # Find control from current state that traverses to goal node.
            # NOTE: could in theory be duplicates! Some ambiquity here...
            control = self.states[node].traverse[path[idx+1]]

            # Find unique edge for state_control pair
            edge = self.edge_collection[StateControl(node, control)]

            # Get state to control mapping for path
            heuristic.update(edge.path)

            # Path is [k_1, k_2, ..., k_M] add k_0 entry state
            heuristic.update([(node, control)])

        return path, heuristic

    def _algorithm(self, current, target):
        """ Implementation of heuristic algorithm. """
        return networkx.shortest_path(self._graph, current, target, 'length')

    def compute_heuristic(self, agent):
        """ Return heuristic for agent. """

        current = agent.current_node
        for target in agent.target_nodes.keys():
            try:
                path_networkx = self._algorithm(current, target)
                path, heuristic = self._path_from_networkx(path_networkx)
                status = PathStatus.FEASIBLE
                break
            except (networkx.exception.NodeNotFound, networkx.NetworkXNoPath) as e:
                path = None
                heuristic = None
                status = PathStatus.INFEASIBLE
        return path, heuristic, status

