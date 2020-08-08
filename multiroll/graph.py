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


from multiroll import *
from .constants import *
from .framework import *
from .coordinate import *
from .agent import *
from .edge import *

class MyGraph(Utils):
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
    def __init__(self, env, env_renderer=None, debug_is_enabled=None):
        self.switch_debug_mode(debug_is_enabled)
        self.visualisation_is_enabled = True

        multiroll.framework.set_env(env)

        multiroll.display.set_env_renderer(env_renderer)

        self._graph = networkx.DiGraph()
        self._graph_complete = networkx.DiGraph()
        self.graph_activity = GraphActivity.ZERO

        self._initialise_agents()
        self._initialise_graph()

    def _locate_agents_in_graph(self):
        for agent in self.agents.values():
            agent.initialise()
            agent.find_railway_target()  #initialise()
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
                # Possible to have multiple edges from one state leading to another edge
                # This can lead to mismatched translation of networkx path tuples to
                # corresponding edge_container_ids
                # print('duplicate edges encountered')
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

    def _initialise_railway(self):
        """ Defines all railway coordinates with unique railway ID. """
        env_railway = numpy.nonzero(self.grid)
        id_railway = -1
        for r, c in zip(*env_railway):
            id_railway += 1
            coordinate = Coordinate(r, c)
            CoordinateContainer(id_railway, coordinate)

    def _create_graph(self, consider_vote=True):
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
        if not consider_vote:
            self._graph_complete.clear()
            for edge_container in self.edges.values():
                edges = edge_container.get_edges(consider_vote=False)
                for edge in edges:
                    self._graph_complete.add_edge(*edge.pair, length=edge.length)
            self._graph = self._graph_complete.copy()
            return
            # path = networkx.shortest_path(self._graph_complete, weight='length')
            # TODO: testpath = networkx.all_shortest_path(self._graph_complete
            # TODO: remove plotting -> add all_path_computation and visualise output
            return
        # TODO: in order to reset the graph simply copy from the shadow (self._graph_complete)
        # NOTE: remove subsequent
        self._graph.clear()
        for edge_container in self.edges.values():
            edges = edge_container.get_edges(consider_vote=consider_vote)
            for edge in edges:
                self._graph.add_edge(*edge.pair, length=edge.length)

    def _update_agents_heuristics(self, optimal=True):
        """ Compute shortest path for each agent. """
        for agent in self.agents.values():
            agent.reset_path()
            self.shortest_path(agent)

    def _conduct_vote(self):
        """ Execute voting on all edges in agent's path.

            This is only for initialisation.

        """
        #for edge_container in self.edges.values():
        #    # NOTE: this triggers UNVOTED -> None in vote_edges
        #    edge_container.reset_vote()

        for agent in self.agents.values():
            agent.vote_edges()

    def _initialise_graph(self):
        self._initialise_railway()
        self._initialise_edges()
        self._locate_agents_in_graph()
        self.debug('Initialise graph')
        self._create_graph(consider_vote=False)
        self._update_agents_heuristics(optimal=True)

        #self._conduct_vote()
        #self.debug('Recompute heuristics with direction constraints')
        #self._create_graph(consider_vote=True)
        #self._update_agent_heuristics()

    def _initialise_agents(self):
        """ Parse flatland metrics from agents. """
        for agent_id, agent in enumerate(self.env.agents):
            self.agents[agent_id] = AgentContainer(agent_id, agent)

    def _all_total_shortest_path(self):
        """Return all shortest path for all edges. """
        s = networkx.all_shortest_path(self._graph_complete)

    def _convert_path_to_edge_containers(self, path):
        """ """
        pass

    def _shortest_path_look_up(self, start, goal):
        """ Parse arguments to networkx implementation. """
        return networkx.shortest_path(self._graph, start, goal, 'length')

    def _shortest_path(self, start, goal):
        """ Parse arguments to networkx implementation. """
        return networkx.shortest_path(self._graph, start, goal, 'length')

    def shortest_path(self, agent):
        """ Update heuristic for agent with agent_id. """
        def agent_text():
            return 'Agent{0}: '.format(agent.id), agent.status,
        current = agent.current_node
        timestamp = time.time()
        for target in agent.target_nodes.keys():
            try:
                sp = self._shortest_path(current, target)
                agent.mode = AgentMode.ACTIVE
                agent.update_path(sp)
                print(sp)
                debug_message = 'SUCCESS! '
                debug_message += '{} '.format(target)
                debug_message += '({:.4}s)'.format(time.time() - timestamp)
                self.debug(agent_text(), debug_message)
                return
            except (networkx.exception.NodeNotFound, networkx.NetworkXNoPath) as e:
                # TODO: consider that agent might be able to use 
                #       existing heuristic as path
                agent.status = AgentStatus.INFEASIBLE_PATH
                agent.mode = AgentMode.STALE
                self.debug(agent_text(), ' ERROR! \n\t', e)
        debug_message = 'Total failure of path comptutation! '
        debug_message += '{} '.format(target)
        debug_message += '({:.4}s)'.format(time.time() - timestamp)
        self.debug(agent_text(), agent.status, debug_message)
        # NOTE: with vote disabled this graph is complete at all time
        print('Infeasible for given path: will never find a target??')
        raise RuntimeError()

    # NOTE: VOTE
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

    # NOTE: VOTE - 
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

    def update_agent_states(self):
        """ Update each agent state with most recent flatland states. """
        for agent in self.agents.values():
            agent.update()

    def visualise(self):
        """ Call display utility methods and visualise metrics and states. """
        if self.visualisation_is_enabled:
            multiroll.display.show_agents(self.agents.values())
            multiroll.display.show()


if __name__ == "__main__":
    print('Graph - Testbed')
    env.reset()
    env_renderer.reset()

    for step in range(500):
        env.step(dict((a,0) for a in range(env.get_num_agents())))

    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)
    input('press to close')

