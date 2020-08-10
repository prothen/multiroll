#!/bin/env python
"""

    Author: Philipp Rothenhäusler, Stockholm 2020

"""

import flatland


from .constants import *
from .framework import *
from .coordinate import *


class AgentContainer(Utils):
    """ Get subset of metrics from flatland environment.

        Note:
            Defines agent interface to flatland agent.

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
        self.path_status = PathStatus.NONE

        self.target = Coordinate(*agent.target)
        # Register target states to request railway coordinate as node in graph
        self.targets[self.target] = None

        # Graph nodes that satifsy target coordinate (find_railway_target)
        self.target_nodes = None

        # List of nodes sorted along path
        self.path = list()
        # Dictionary with State to ControlDirection
        self.controller = dict()

    @property
    def current_node(self):
        """ Return next node from path or otherwise state.

            Warning:
                Read-only and an unpopulated path will cause invalid indexing.

        """
        if self.status == AgentStatus.ON_PATH:
            return self.path[0]
        return self.state

    def edge_container_ids(self):
        return [e.id for e in self.path_edge_containers]

    def find_railway_target(self):
        """ Set agents target CoordinateContainer and its admissable states.

            Note
                Requires the self.railway to be initialised through Graph
        """
        target_container = self.railway[self.target]
        self.target_nodes = target_container.valid_states

    def reset_path(self):
        """ Reset path its controller and the AgentStatus. """
        self.path = list()
        self.controller = dict()
        self.path_status = PathStatus.NONE

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
            path_controller = edge_container.path[edge_direction]
            self.controller.update(path_controller)
            self.next_node = edge_container.goal_state[edge_direction]
            self.path = [goal_state]
            self.status = AgentStatus.ON_PATH
            return
        self.current_node = self.state
        self.status = AgentStatus.ON_NODE

    def set_controller(self, path, controller):
        """ Update edge_id and state to control dictionary. """
        # TODO: if ON_NODE -> get edge that leads to path
        self.path = path
        self.controller.update(controller)
        # TODO: Refactor and clean up (not necessary)
        self.controller.update([(path[-1], ControlDirection(Control.S, None))])
        self.path_status = PathStatus.FEASIBLE

    def get_control(self):
        """ """
        return self.controller[self.state].control


    def update(self):
        """ Update agent state with flatland environment state.

            Note:
                Sets the agent.status that coordinates the current_node
                property access.

        """
        if self.state in self.targets:
            print('REACHED goal')
            self.controller.update([(self.state, ControlDirection(Control.S, None))])
            #self.contr
        #if self.states[self.state].coc.id == self.
        if not self._agent.status == flatland.envs.agent_utils.RailAgentStatus.ACTIVE:
            return
        a = self._agent
        d = a.direction
        (r, c) = a.position
        d = Direction(a.direction)
        self.state = State(r, c, d)

        if not self.path_status == PathStatus.INFEASIBLE:
            self.status = AgentStatus.ON_PATH
            if self.state in self.path:
                self.path = self.path[self.path.index(self.state)+1:]
                self.status = AgentStatus.ON_NODE
            if len(self.path) == 0:
                self.path_status = PathStatus.INFEASIBLE
                self.status = AgentStatus.NONE

