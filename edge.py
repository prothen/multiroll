#!/bin/env python

from constants import *
from framework import *
from coordinate import *
from agent import *

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
    """
    def __init__(self, ID, debug_is_enabled=True):
        self.id = ID
        self.active_agents = dict()
        self.switch_debug_mode(debug_is_enabled)

        self.vote = 0
        self.vote_status : VoteStatus = VoteStatus.UNVOTED

        # EdgeDirection key with goal_state value
        self.goal_state = dict()
        # DirectionType key and common path values (2 entries)
        self.path = dict()
        # State keys and EdgeDirection values
        self.length = None
        # Store previous direction to improve edge aditions
        self._previous_direction = None

        # store forward and backward edges under EdgeDirection Key
        self._edge_registry = dict()
        # collection of container edges with key being EdgeDirection
        self._edges = dict([(None, self._edge_registry)])
        self._edges[EdgeDirection.FORWARD] = dict()
        self._edges[EdgeDirection.BACKWARD] = dict()

        self._edge_direction = dict()
        self._agent_registry = dict([(e, list()) for e in EdgeDirection])
        self._edge_actions = dict()

        # State to cell id relative to edge (edge switch indicator for agent)
        self.state2progress = dict()
        # State to edge direction for container (localise on edge)
        self.state2direction = dict()

    def _reset_vote(self):
        """ Reset vote and allow all edge directions to be used. """
        self.vote = 0
        self.vote_status = VoteStatus.UNVOTED
        self.previous_direction = None

    def _vote_result(self):
        """ Return None for unvoted or undecided. """
        if not self.vote_status == VoteStatus.UNVOTED:
            if self.vote >= 0:
                return EdgeDirection.FORWARD
            return EdgeDirection.BACKWARD
        return None

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
            self.state2direction[state] = edge_direction
        self.length = len(path)

    def get_edge_updates(self):
        """ Return vote-preferred edges for corresponding actions.

            Note:
                Skip already elected edge changes.
        """
        if self.vote_status & VoteStatus.ELECTED:
            return dict()

        self.vote_status |= VoteStatus.ELECTED
        return self.edge_action

    def get_edges(self, consider_vote=True):
        """ Return available edges under evaluated vote.

            Note:
                If no agent has claimed interest, all edges are returned.
        """
        print('EC{}'.format(self.id), '(Vote{})'.format(self.vote))
        if consider_vote:
            self.evaluate_vote()
            return self._edges[self._vote_result()].values()
        return self._edge_registry.values()

    def evaluate_vote(self):
        """ Evaluate voting and stage edges for removal and addition.

            Note:
                In order to reset voting and activating all edges use
                reset_vote().
        """
        direction = self._vote_result()
        direction_reverse = EdgeDirection.reverse(direction)
        if self._previous_direction == direction_reverse:
            self._edge_action[EdgeActionType.ADD] = self._edges[direction]
            self._edge_action[EdgeActionType.REMOVE] = self._edges[direction_reverse]
            self._previous_direction = direction
            return

    def parse_agent_vote(self, state, agent):
        """ Register interest to use an edge in certain direction.

            Note:
                Agents in AgentMode.STALE are casted to
                    agent.mode = AgentMode.EXPLORING 
                once deactivated edge becomes reenabled. (see exit trigger)
        """
        # print('Agent vote for {}'.format(self.id), '\n\t', agent.edge_container_ids())
        ids = agent.edge_container_ids()
        edge_direction = self.state2direction[state]
        self.vote += edge_direction
        self._agent_registry[edge_direction].append(agent.id)
        self.vote_status = VoteStatus.VOTED

    def force_vote(self, edge_direction, agent):
        """ Enforce directional reservation for agents starting on edge. """
        self._reset_vote()
        self.vote += edge_direction
        self._agent_registry[edge_direction].append(agent.id)
        self.voted = True

    def enter(self, state, agent_id):
        """ Conduct entry procedure for agent from state.

            Note:
                The global priority_dict helps in selecting
                promising candidates for the rollout.

            Todo:
                Register agents globally to improve the priority
                dict for rollout.
        """
        self.active_agents[agent_id] = self.agents[agend_id]
        #   and reference to agent's traverse metrics
        # self.priority_dict[edge.priority][agent_id] = agent_container

    def get_agent_progress(self, agent_id, state):
        """ Update agent_id edge progress and return eta in cell count.

            Note:
                On zero progress the entry trigger is invoked and on
                estimated time of arrival (ETA) with eta being zero the
                exit trigger.
        """
        progress = self.state2progress[state]
        if not progress:
            self.enter(agent_id)
        eta = self.length - progress -1
        if not eta:
            self.exit(agent_id)
            #raise RuntimeError()
        return eta

    def _update_occupancy(self):
        """ Conduct freeing routine when last active agent leaves edge.

            Todo:
                Reverse direction for affected agent retrieval.

        """
        if len(self.active_agents) == 0:
            direction = self._vote_result()
            direction_reverse = EdgeDirection.reverse(direction)
            self._edge_action[EdgeActionType.ADD] = self._edges[direction_reverse]
            self._reset_vote()
            # TODO: activate AgentMode.RECOMPUTE_REQUEST -> 8 (unique) & |
            #       -> in graph test all agents
            # TODO; use global dictionary for reactivation schedule
            # TODO; add callback for id in each agent
            #       - from graph fetch pending agent requests
            # update graph
            # make 
            # compute path
            # vote
            # recompute / make infeasible agents stale
            for agent_id in self._agent_registry[direction_reverse]:
                # TODO: -> call this in agent.vote_path for each element
                self.agents[agent_id].vote_edge_container(self.id)
            self.evaluate_vote
            # TODO: activate all of this container edges
            #       compute heuristic for all agents in registry
            #       check if agents disagree -> need to recompute their edges
            # TODO; store for each agent -> edge_container_id to node 
            # TODO: add method to vot for specific edge_container
            # revote from all agents in registry 
            # since some agents could be on suboptimal route that
            # prefers this to be in a different direction than initial vote
            edge_direction = EdgeDirection.reverse(self._vote_result())
            self._reset_vote()
            for agent_id in self._agent_registry[edge_direction]:
                self.agents[agent_id].update_edge_availability(self.id)

    def exit(self, agent_id):
        """ Remove agent from active_agents and priority_dict. """
        #prio = self.active_agents[agent_id].priority
        #self.priority_dict[prio].pop(agent_id, None)
        self.active_agents.pop(agent_id, None)
        self._update_occupancy()

