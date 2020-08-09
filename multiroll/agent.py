#!/bin/env python

import flatland

from .constants import *
from .framework import *
from .coordinate import *


class AgentContainer(Utils):
    """ Get subset of metrics from flatland environment.

        Note:
            Defines agent interface to flatland agents.

        TODO:
            - architecture design
            - interfaces to simulator
    """
    def __init__(self, ID, agent, debug_is_enabled=None):
        self.id = ID
        self._agent = agent
        self.switch_debug_mode(debug_is_enabled)

        a = self._agent
        (r, c) = a.initial_position
        d = Direction(a.initial_direction)
        self.state = State(r, c, d)
        self.status = AgentStatus.INITIALISED

        self.target = Coordinate(*agent.target)
        # Register target states to request railway coordinate as node in graph
        self.targets[self.target] = None

        # Graph nodes that satifsy target coordinate (find_railway_target)
        self.target_nodes = None
        # Defines next node available for path decision
        self.next_node = None

        # TODO:
        #       should be ControlDirection to facilitate simulation
        self.heuristic = dict()

    def edge_container_ids(self):
        return [e.id for e in self.path_edge_containers]

    def find_railway_target(self):
        """ Set agents target CoordinateContainer and its admissable states.

            Note
                Requires the self.railway to be initialised through Graph
        """
        target_container = self.railway[self.target]
        self.target_nodes = target_container.valid_states
        self.status |= AgentStatus.HAS_TARGET

    def reset_path(self):
        """ Reset path and the AgentStatus. """
        self.heuristic = dict()
        self.path_edge_containers = list()
        self.status = AgentStatus.INFEASIBLE_PATH
        self.locate()

    def locate(self):
        """ Locate agent in graph and Find next search node.

            Note:
                If on edge use its path as heuristic.
        """
        self.reset_path()
        state_container = self.states[self.state]
        if not state_container.type & StateType.NODE:
            # TODO: Consider testing all edges! Possible ambiguity in networkx
            edge_container = self.edges[state_container.edges[0]]
            edge_direction = edge_container.state2direction[self.state]

            goal_state = edge_container.goal_state[edge_direction]
            self.heuristic.update(edge_container.path[edge_direction])
            self.next_node = edge_container.goal_state[edge_direction]
            self.path.append(edge_container)
            return
        self.current_node = self.state

    def update(self):
        """ Update agent state with flatland environment state. """
        if not self._agent.status == flatland.envs.agent_utils.RailAgentStatus.ACTIVE:
            print('SKIPPED agent state update since non-active --> RTD STALE?')
            return
        a = self._agent
        d = a.direction
        (r, c) = a.position
        d = Direction(a.direction)
        self.state = State(r, c, d)

        # update state
        # TODO:
        #       if agent on path_node[0] -> del path_node[0]
        #       set AgentMode on NODE
        # 
