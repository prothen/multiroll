#!/usr/bin/env python
"""
    Multiagent rollout implemented on flatland environment.


    Note:
        This module expects to be initialised with

            import graph
            graph.set_env(env)
            ...
            graph.MyGraph()

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


from multiroll.scheduler import *

from .constants import *
from .framework import *
from .coordinate import *
from .agent import *
from .edge import *


class MyGraph(Utils, multiroll.graph.MyGraph):
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

    def _initialise_graph(self):
        super()._initialise_graph()

        # self._conduct_vote()
        # self.debug('Recompute heuristics with direction constraints')
        # self._create_graph(consider_vote=True)
        # self._update_agent_heuristics()

    def _create_graph_voted(self):
        """ Initialise networkx graph with all edges.

            Note:
                This method is for clean recreation of complete graph with
                voted edges.

                It is recommended to rely on _update_graph automation and
                automated edge reactivation on agent exiting the edge_container.

                Recommended to first run vote edges and update edge_containers
                to remove deadlocks.

                Consider_vote flag allows to select only prioritised edges from
                each edge_container. (unidirectional section use)

            TODO:
                Remove edge_containers from voting, if active_agents on edge
                (e.g.: from init)
        """
        self._graph.clear()
        for edge_container in self.edges.values():
            edges = edge_container.get_edges(consider_vote=consider_vote)
            for edge in edges:
                self._graph.add_edge(*edge.pair, length=edge.length)

    def _conduct_vote(self):
        """ Execute voting on all edges in agent's path.

            This is only for initialisation.

        """
        #for edge_container in self.edges.values():
        #    # NOTE: this triggers UNVOTED -> None in vote_edges
        #    edge_container.reset_vote()

        for agent in self.agents.values():
            agent.vote_edges()

    def _update_graph_edges(self):
        """ Update graph by adding and removing voted edges graph. """
        pop_list = list()
        for edge_container in self.edge_reactivation.values():
            new_edges = edge_container.get_edge_updates()
            for action, edges in new_edges.items():
                if not len(edges):
                    continue
                if action == EdgeActionType.ADD:
                    for edge in edges.values():
                        self._graph.add_edge(*edge.pair, length=edge.length)
                    continue
                for edge in edges.values():
                    self._graph.remove_edge(*edge.pair, length=edge.length)
            pop_list.append(edge_container.id)
        for pop in pop_list:
            self.edge_reactivation.pop(pop, None)
        return

    def _is_agent_exploring(self, agent):
        """ Test if agent has interest in recently reactivated edge. """
        # print('Agent{}'.format(agent.id), agent.mode, agent._agent.status)
        # TODO: add agent.status, agent_id from flatland)
        # TODO: show debug output from rail_env
        if (agent.mode == AgentMode.STALE or
            agent._agent.status == flatland.envs.agent_utils.RailAgentStatus.DONE_REMOVED):
            # print('skip {}'.format(agent.id))
            return False # NOTE: Leave moving trains untouched
        self.graph_activity |= GraphActivity.AGENT_ACTIVE
        if agent.mode == AgentMode.EXPLORING:
            self._update_graph_edges()
            self.debug_is_enabled = True
            self.shortest_path(agent)
            self.debug_is_enabled = False
        return True
        #if agent.mode == AgentMode.STALE:
        #    return

    # NOTE: Final placement under rollout.py
    def controls(self):
        """ Return control dictionary for all agents. """
        controls = dict()
        # self.graph_activity = GraphActivity.ZERO
        # self._update_agent_heuristics(optimal=True)
        i = 0

        for agent in self.agents.values():
            print('Agent', agent.id, ': ', agent._agent.status, ' ', agent.state)
            if agent._agent.status == flatland.envs.agent_utils.RailAgentStatus.DONE_REMOVED:
                controls[agent.id] = Control.S
                i+=1
                continue
            #if agent.mode == AgentMode
            #print('Agent'.agent.id,': ', agent.mode)
            agent.set_control(controls)
            # TODO: check how flatland parses controls and 
            #       whether agent_id still active
            #if self._is_agent_exploring(agent):
            #    agent.set_control(controls)
            #    continue
            #if self.graph_activity == GraphActivity.ZERO:
            #    print('Stale graph')
            #    controls[agent.id] = self.states[agent.state][0].control
        #print(controls)
        print('Agents in goal:', i)
        return controls
        print('GraphActivity: ', self.graph_activity)
        if self.graph_activity == GraphActivity.ZERO:
            self._graph = self._graph_complete.copy()
            #self._create_graph(consider_vote=False)
            self._update_agent_heuristics(optimal=True)
            self._conduct_vote()
            #self.debug('Recompute heuristics with direction constraints')
            self._create_graph(consider_vote=True)
            self._update_agent_heuristics()
            #import random
            #agent = random.choice(list(self.agents.values()))
            #controls_i = self.states[agent.state].controls
            # print('select from controls')
            #print(controls_i)
            #controls[agent.id] = random.choice(controls_i).control
            #for agent in self.agents.values():
            #    agent.mode = AgentMode.EXPLORING
        for agent in self.agents.values():
            #continue
            if agent.mode == AgentMode.ACTIVE:
                #rollout.display.show_states([agent.state], Color.RED, Dimension.RED)
                #display.show_states([agent.state], Color.STATE, Dimension.STATE)
                #display.show()

                sc = self.states[agent.state]
                coc = sc.coc
                co = coc.coordinate
                control_bits = self.grid[co]

                #print('State: {}'.format(agent.state))
                #print('GRID: {}'.format(self.grid[co]))
                #print('BITS', bin(control_bits))
                #print('EDGES: {}'.format(agent.edge_container_ids()))
                #e_id = agent.edge_container_ids()[0]
                #ec = self.edges[e_id]
                #print(ec.path)
                #print('AGENT{}: '.format(agent.id), controls[agent.id])
                #print('Available COC: {}'.format(coc.controls))
                #print('Available SC: {}'.format(self.states[agent.state].controls))
        return controls


if __name__ == "__main__":
    print('Graph - Testbed')

