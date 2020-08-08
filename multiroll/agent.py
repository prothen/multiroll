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
        # TODO: maybe rename to path_status
        self.status = AgentStatus.INITIALISED
        # NOTE: should also consider whether any interaction with it is necessary
        # -> see DONE_REMOVE, DONE
        self.mode : AgentMode = AgentMode.EXPLORING

        # Note: initialised in graph (locate_agents_in_graph)
        self.target = Coordinate(*agent.target)
        # To be documented: Usage and necessity?
        self.target_container = None
        # Add key for this agents target coordinate to global targets
        # NOTE: Used in decision of CoordinateContainer
        #       whether coordinate states become nodes
        self.targets[self.target] = None

        # Define possible search nodes for path computation
        self.target_nodes = None
        # Defines next node available for path decision
        self.current_node = None

        # TODO: when initialised?
        # Agent specific goal states that satisfy target coordinate
        self.tc : StateContainer = None
        self.target_edges = list()

        # speed = a.speed_data['speed']
        # import math; self.speed = math.ceil(1/speed)
        self.path = list()
        self.path_edge_containers = list()
        self.path_edge_container_ids = dict()
        self.path_nodes = list()
        self.heuristic = dict()

    def edge_container_ids(self):
        return [e.id for e in self.path_edge_containers]

    # LEGACY
    def initialise(self):
        """ Fetch current states and update targets.

            Note:
                Called after MyGraph has initialised railway.

        """
        self.target_container = self.railway[self.target]
        self.target_nodes = self.target_container.valid_states

    def find_railway_target(self):
        """ Set the agents railway target coordinate and possible states.

            Note:
                Requires the railway in GlobalContainer to be initialised.

            Todo:
                Consider try except or trust that target is on railway in GlobalCoordinate

        """
        self.target_container = self.railway[self.target]
        self.target_nodes = self.target_container.valid_states
        self.status |= AgentStatus.HAS_TARGET

    # NOTE: VOTE
    def update_edge_availability(self, edge_container_id):
        """ Test if this agent is interested to hear about the reactivation of
            a previously voting-related denied edge.
        """
        if not self.mode == AgentMode.ACTIVE:
            self.mode = AgentMode.EXPLORING

    def reset_path(self):
        """ """
        # Here first unregister from the whole previous path
        self.unregister_edges()
        self.heuristic = dict()
        self.path = list()
        self.path_edge_containers = list()
        self.path_nodes = list()
        self.heuristic = dict()
        self.status = AgentStatus.INFEASIBLE_PATH
        self.locate()

    def locate(self):
        """ Locate agent in graph and Find next search node.

            Note:
                If on edge use its path as heuristic.
        """
        state_container = self.states[self.state]
        if not state_container.type & StateType.NODE:
            edge_container_id = state_container.edges[0]
            edge_container = self.edges[edge_container_id]
            edge_direction = edge_container.state2direction[self.state]

            edge_container.force_vote(edge_direction, self)
            goal_state = edge_container.goal_state[edge_direction]
            self.heuristic.update(edge_container.path[edge_direction])
            self.current_node = edge_container.goal_state[edge_direction]
            self.path_edge_containers.append(edge_container)
            # NOTE: debug -> remove for production
            # print('Edge direction initialised: {}'.format(edge_direction))
            # print('On Edge container: {}'.format(edge_container_id))
            return
        self.current_node = self.state

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
        self.path_nodes = path
        for idx, node in enumerate(path):
            if idx == len(path)-1:
                continue
            control = self.states[node].traverse[path[idx+1]]
            edge = self.edge_collection[StateControl(node, control)]
            edge_path = edge.path
            edge_container = self.edges[edge.container_id]
            self.path_edge_containers.append(edge_container)
            self.path_edge_container_ids[edge.container_id] = edge_container
            self.heuristic.update(edge_path)
            self.heuristic.update([(node, control)])
            self.status = AgentStatus.FEASIBLE_PATH
        self.register_edges()

    # NOTE: VOTE
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

    # NOTE: VOTE
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

    # NOTE: VOTE
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
        # print('Initial target: {}'.format(self.target))
        # print('Flatland target: {}'.format(self._agent.target))

    def update_edge_progress(self, edge_container):
        """ Return the amount of cells remaining after current state. 

            Note:
                This implicitly switches to the next edge_container_id and
                in get_agent_progress tests if edge availability changes.

        """
        eta = edge_container.get_agent_progress(self.id, self.state)
        if not edge_container.get_agent_progress(self.id, self.state):
            self.path_edge_containers.pop(0)

    # NOTE: LEGACY -> handled by rollout
    def set_control(self, controls):
        """ Update control dictionary and update active linked edge_containers.

            Note:
                Condition actions (skipping on flatland states)
                see flatland.envs.agent_utils.RailAgentStatus

        """
        #print('Agent', self.id, ': has', self.status)
        if not self.status == AgentStatus.FEASIBLE_PATH:
            #print('ID: ', self.id, ' No Path: Stopping!')
            controls[self.id] = Control.S
            return
        controls[self.id] = self.heuristic[self.state].control
        self.update_edge_progress(self.path_edge_containers[0])
        #print(self.heuristic)
        #print('Current control:', self.heuristic[self.state].control)
        #print(controls[self.id])
        # print('Agent{}: '.format(self.id), '\nPath-IDs:{}'.format(self.edge_container_ids()))


class AgentTraverse(GlobalContainer):
    """ Utility class for maintaining overview of useful metrics. """

    def __init__(self, agent_id, edge):
        self.id = agent_id
        # Fetch agent container
        self.agent = self.agents[agent_id]
        # Traversal priority from edge
        self.priority = edge.priority
        self.edge = edge
        self.speed = self.agent.speed

