#!/bin/env python


import flatland


from multiroll.scheduler import *

from .constants import *
from .framework import *
from .coordinate import *


class AgentContainer(Utils, multiroll.agent.AgentContainer):
    """ Get subset of metrics from flatland environment.

        Note:
            Defines agent interface to flatland agents.

        TODO:
            - architecture design
            - interfaces to simulator
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_path(self):
        """ """
        self.unregister_edges()
        super().reset_path()

    def update_path(self, path):
        """ Use path with list of nodes to receive agent heuristics.

            Note:
                Path format defined from networkx with corresponding
                edge node entries.

            Todo:
                Debug the conversion to edge_ids

                - Convert networkX path in graph to edge_container ref

                - Only parse edge_containers here
        """
        super().update_path(path)
        self.register_edges()

    def update_edge_availability(self, edge_container_id):
        """ Test if this agent is interested to hear about the reactivation of
            a previously voting-related denied edge.
        """
        if not self.mode == AgentMode.ACTIVE:
            self.mode = AgentMode.EXPLORING

    def vote_edges(self):
        """ Register interest in using edge_container_ids from its path. 

            Note:
                store initial heuristic (isolated optimal shortest path) in
                attribute and recover

        """
        if not self.status == AgentStatus.FEASIBLE_PATH:
            print('Agent has been asked to vote but does not have feasible path')
            #raise RuntimeError()
        for edge_container_id, node in zip(self.edge_container_ids(),
                                           self.path_nodes):
            self.edges[edge_container_id].parse_agent_vote(node, self)

    def unregister_edges(self):
        """ Unregister to all  edge_containers on path.

            Note:
                This is done once whenever the agent receives a
                new path reset via reset_path.

            Todo:
                Avoid collisions while maintaining dynamic
                reconfigurability for freed edge_containers.
        """
        for edge_container in self.path_edge_containers:
            edge_container.unregister_agent(self.id)

    def register_edges(self):
        """ Register to all  edge_containers on path.

            Note:
                This is done once whenever the agent receives a
                new path via update_path.

            Todo:
                Avoid collisions while maintaining dynamic
                reconfigurability for freed edge_containers.
        """
        for edge_container in self.path_edge_containers:
            edge_container.register_agent(self.id)

    def update(self):
        """ Update agent state with flatland environment state. """
        if not self._agent.status == flatland.envs.agent_utils.RailAgentStatus.ACTIVE:
            return
        a = self._agent
        d = a.direction
        (r, c) = a.position
        d = Direction(a.direction)
        self.state = State(r, c, d)


class AgentTraverse(Utils):
    """ Utility class for maintaining overview of useful metrics. """

    def __init__(self, agent_id, edge):
        self.id = agent_id
        # Fetch agent container
        self.agent = self.agents[agent_id]
        # Traversal priority from edge
        self.priority = edge.priority
        self.edge = edge
        self.speed = self.agent.speed

