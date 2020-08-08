#!/bin/env python


from .constants import *
from .framework import *
from .coordinate import *
from .agent import *


class EdgeContainer(Utils):
    """ Edge related metrics and occupancy trackers.

        Note:
            Set default debug_is_enabled=None to fetch
            global debug_mode and set to True to activate
            debug for all EdgeContainers.

        Note:
            After vote evaluation edge dict references are added to
            edge_action under EdgeActionType key.
            The entries are then fetched using get_edge_updates(),
            which is conditioned on note(self.vote_status & ELECTED) and
            on completion sets | VoteStatus.ELECTED

        Todo:
            1.
                Register agents and their planned entry step
                -> If agent registers and no vote done -> force switch

            2.
                Add collision matrix with agent steps for
                global N prediction steps along path length M
                -> matrix &operator should yield zero for collision free

            3.
                AgentContainer -> get_control -> update current_edge -> move_agent
                    - If edge exit, select next edge_id from path_id

            4. Voting has matured to a point that deserves a separate object.

            5. Mange voting result globally through dictionary on edge_container 
                -> move self.vote to a dict under GlobalContainer
    """
    def __init__(self, ID, debug_is_enabled=True):
        self.id = ID
        self.switch_debug_mode(debug_is_enabled)

        # EdgeDirection key with goal_state value
        self.goal_state = dict()
        # DirectionType key and common path values (2 entries)
        self.path = dict()
        # State keys and EdgeDirection values
        self.length = None
        # Store previous direction to improve edge aditions
        self._previous_direction = None

        # Store forward and backward edges under EdgeDirection Key
        self._edge_registry = dict()
        # collection of container edges with key being EdgeDirection
        self._edges = dict([(None, self._edge_registry)])
        self._edges[EdgeDirection.FORWARD] = dict()
        self._edges[EdgeDirection.BACKWARD] = dict()

        self._edge_direction = dict()
        # Collect agent interest in corresponding direction
        self._agent_registry = dict([(e, dict()) for e in EdgeDirection])
        self._edge_action = dict()

        # State to cell id relative to edge (edge switch indicator for agent)
        self.state2progress = dict()
        # State to edge direction for container (localise on edge)
        self.state2direction = dict()

    def add_edges(self, edges, backward=False):
        """ Add edges according to EdgeDirection to dict. """
        edge_direction = self._get_direction(backward)
        for edge in edges:
            self._edge_registry[edge.pair.vertex_1] = edge
            self._edges[edge_direction][edge.pair.vertex_1] = edge
            self._edge_direction[edge.pair.vertex_1] = edge_direction

    def add_states(self, ingress_states, path, backward=False):
        """ Store common path in attribute according to EdgeDirection.

            Note:
                The ingress states are a dictionary of keys with states
                that share the same traversability direction
                and have values with ControlDirection.
        """
        edge_direction = self._get_direction(backward)
        self.goal_state[edge_direction] = path[-1].state
        self.path[edge_direction] = path[:-1]
        for progress, StateControl in enumerate(path):
            self.state2progress[StateControl.state] = progress
            self.state2direction[StateControl.state] = edge_direction
        for state in ingress_states.keys():
            self.state2progress[state] = 0
            self.state2direction[state] = edge_direction
        self.length = len(path)

    def get_edges(self, consider_vote=True):
        """ Return available edges under evaluated vote.

            Note:
                If no agent has claimed interest, all edges are returned.
        """
        if consider_vote:
            self.evaluate_vote()
            return self._edges[self._vote_result()].values()
        return self._edge_registry.values()

    def get_agent_progress(self, agent_id, state):
        """ Update agent_id edge progress and return eta in cell count.

            Note:
                If the returned eta is zero the agent calling this method
                will pop a edge_container_id from his path and move to the
                next.

            Note:
                On zero progress the entry trigger is invoked and on
                estimated time of arrival (ETA) with eta being zero the
                exit trigger.
        """
        a = self.agents[agent_id]
        sc = self.states[a.state]
        progress = self.state2progress[state]
        if not progress:
            self.enter(state, agent_id)
        eta = self.length -1 - progress
        return eta

