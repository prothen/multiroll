#!/usr/bin/env python
"""
    Multiagent rollout implemented on flatland environment.


    Note:
        This module expects to be initialised with the 
        interfacing environment

            import graph
            ...
            graph.Graph(env)

        before usage of any subsequently defined classes.
"""


import networkx

from multiroll import *
from .constants import *
from .framework import *
from .coordinate import *
from .agent import *
from .edge import *


class ShortestPath(multiroll.graph.Graph):
    """Container for graph related actions.

        Note:
            Update active edges or compute shortest path.

        Note: / TODO:
            Maybe easier to maintain two graphs (one complete and a subset 
            dynamically updated)

        Todo: 
            - Use all_shortest_path computation 
            - Use graph_complete mirror and 
                - use graph_complete.copy() to reset dynamic?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _path_from_networkx(self, path):
        """ Convert the networkX tuples into edge_container references. """
        path_nodes = list()
        for idx, node in enumerate(path):
            if idx == len(path)-1:
                continue
            # Find control from current state that traverses to goal node.
            # NOTE: could in theory be duplicates! Some ambiquity here...
            control = self.states[node].traverse[path[idx+1]]

            # Find unique edge for state_control pair
            edge = self.edge_collection[StateControl(node, control)]

            # Get state to control mapping for path
            heuristic = edge.path

            # Path is [k_1, k_2, ..., k_M] add k_0 entry state
            heuristic.update([(node, control)])

            # Find its edge_container through the edge.id
            edge_container = self.edges[edge.container_id]
            path_edges.append(edge_container)

        # TODO
        print(path)
        print('is a list?')
        raise RuntimeError()
        return path, heuristic

    def _algorithm(self, current, target):
        """ Implementation of heuristic algorithm.

            Todo:
                Use all_shortest_path

        """
        return networkx.shortest_path(self._graph, current, target, 'length')

    def compute_heuristic(self, agent):
        """ Return heuristic for agent. """

        current = agent.current_node
        for target in agent.target_nodes.keys():
            try:
                path_networkx = self._algorithm(current, target)
                path, heuristic = self._path_from_networkx(path_networkx)
                status = AgentStatus.FEASIBLE_PATH
            except (networkx.exception.NodeNotFound, networkx.NetworkXNoPath) as e:
                path = None
                heuristic = None
                status = AgentStatus.INFEASIBLE_PATH
        return path, heuristic, status

