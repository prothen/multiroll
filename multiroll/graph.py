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


import copy
import enum
import time
import numpy
import pandas
import ctypes
import networkx
import itertools
import matplotlib
import collections
from typing import Optional


import flatland


from multiroll import *
from .constants import *
from .framework import *
from .coordinate import *
from .agent import *
from .edge import *

class Graph(Utils):
    """ Graph implementation based on networkX.

    """
    def __init__(self, env, env_renderer=None, debug_is_enabled=None):
        self.switch_debug_mode(debug_is_enabled)

        set_env(env)
        multiroll.display.set_env_renderer(env_renderer)

        self.graph_activity = GraphActivity.ZERO
        self._graph = networkx.DiGraph()

        self._initialise_agents()
        self._initialise_graph()

    def _initialise_agents(self):
        """ Parse flatland metrics from agents. """
        for agent_id, agent in enumerate(self.env.agents):
            self.agents[agent_id] = AgentContainer(agent_id, agent)

    def _initialise_graph(self):
        self._initialise_railway()
        self._initialise_edges()
        self._locate_agents_in_graph()
        self.debug('Initialise graph')
        self._create_graph()

    def _initialise_railway(self):
        """ Defines all railway coordinates with unique railway ID. """
        env_railway = numpy.nonzero(self.grid)
        id_railway = -1
        for r, c in zip(*env_railway):
            id_railway += 1
            coordinate = Coordinate(r, c)
            CoordinateContainer(id_railway, coordinate)

    def _locate_agents_in_graph(self):
        """ Run agent relocalisation routine. """
        for agent in self.agents.values():
            agent.find_railway_target()
            agent.locate()

    def _is_explored(self, state, control):
        return StateControl(state, control) in self.edge_collection.keys()

    def _edge_ingress_states(self, state, direction):
        """ Find all states at coordinate that lead to the same edge. """
        return self.states[state].coc.direction2states[direction]

    def _reverse_edge_ingress_states(self, path):
        """ Return all states that led to edge from reversed direction. """
        goal_state = path[-1].state
        direction = FlipDirection[path[-2].control.direction]

        return self._edge_ingress_states(goal_state, direction)

    def _find_edge_path(self, ingress_states, edge_container_id):
        """ Return List of StateControl for state and control.

            Note:
                Returns list of (state,control) from state+1 cell until goal.

        """
        path = list()
        state, control = list(ingress_states.items())[0]
        path.append(StateControl(state, control))
        state = Simulator(state, control)

        while not self.states[state].type & StateType.NODE:
            self.states[state].edges.append(edge_container_id)
            control = self.states[state].controls[0]
            path.append(StateControl(state, control))
            state = Simulator(state, control)

        path.append(StateControl(state, ControlDirection(Control.S, None)))
        return path

    def _define_edges_from_path(self, ingress_states, path, 
                                edge_container_id):
        """ Parse entry_states and path into edges. """
        edges = list()
        goal_state = path[-1].state
        for ingress_state, control in ingress_states.items():
            if self._is_explored(ingress_state, control):
                raise RuntimeError()
            edge_id = len(self.edge_collection) + 1
            edge_path = [StateControl(ingress_state, control)] + path

            priority = self.states[goal_state].priority
            pair = Pair(ingress_state, goal_state)
            edge = Edge(pair, priority, edge_path, len(edge_path), 
                        edge_container_id)

            self.edge_collection[StateControl(ingress_state, control)] = edge
            edges.append(edge)
            if goal_state in self.states[ingress_state].traverse.keys():
                # NOTE: Cause failure to investigate if multiple edges to same
                #       goal state. Could overwrite edge in networkX tuples.
                raise RuntimeError()
            self.states[ingress_state].traverse[goal_state] = control

        return edges

    def _initialise_edges(self):
        """ Iterate over nodes and return edge containers. """
        for node in self.nodes.values():
            controls = node.controls
            for control in controls:
                if self._is_explored(node.state, control):
                    continue
                edge_container_id = len(self.edges) + 1
                edge_container = EdgeContainer(edge_container_id)

                # Forward edges
                direction = control.direction
                ingress_states = self._edge_ingress_states(node.state, direction)
                path = self._find_edge_path(ingress_states, edge_container_id)
                edges = self._define_edges_from_path(ingress_states, path,
                                                     edge_container_id)
                edge_container.add_states(ingress_states, path, backward=False)
                edge_container.add_edges(edges, backward=False)

                # Backward edges
                ingress_states = self._reverse_edge_ingress_states(path)
                path = self._find_edge_path(ingress_states, edge_container_id)
                edges = self._define_edges_from_path(ingress_states, path,
                                                     edge_container_id)
                edge_container.add_states(ingress_states, path, backward=True)
                edge_container.add_edges(edges, backward=True)

                self.edges[edge_container_id] = edge_container
            node.edges.append(edge_container_id)

    def _create_graph(self):
        """ Initialise networkx graph with all edges. """
        self._graph.clear()
        for edge_container in self.edges.values():
            edges = edge_container.get_edges()
            for edge in edges:
                self._graph.add_edge(*edge.pair, length=edge.length)
        return


    def visualise(self):
        """ Call display utility methods and visualise metrics and states.

            Todo:
                Add graph visualisation specifics.

        """
        multiroll.display.show_agents(self.agents.values())
        multiroll.display.show()


if __name__ == "__main__":
    print('Graph - Testbed')

