#!/bin/env python


from .constants import *
from .framework import *
from .coordinate import *
from .agent import *


class EdgeContainer(Utils):
    """ Edge related metrics and occupancy trackers. """

    def __init__(self, ID, debug_is_enabled=True):
        self.id = ID
        self.switch_debug_mode(debug_is_enabled)

        # EdgeDirection key with goal_state value
        self.goal_state = dict()
        # DirectionType indexed controllers (State, ControlDirection) pairs
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

    def _get_direction(self, backward):
        """ Return the EdgeDirection for backward argument. """
        if backward:
            return EdgeDirection.BACKWARD
        return EdgeDirection.FORWARD

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

    def get_edges(self):
        """ Return all available edges for this EdgeContainer. """
        return self._edge_registry.values()

