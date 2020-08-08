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
            
            5. Mange voting result globally through dictionary on edge_container 
                -> move self.vote to a dict under GlobalContainer
    """
    def __init__(self, ID, debug_is_enabled=True):
        self.id = ID
        self.switch_debug_mode(debug_is_enabled)

        self.vote = 0
        self.vote_status : VoteStatus = VoteStatus.UNVOTED

        # Actively moving agents
        self.active_agents = dict()
        # Voted agents expressed interest in moving over this section in the future
        self.registered_agents = dict()

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

    # NOTE: VOTE
    def _vote_result(self):
        """ Return None for unvoted or undecided. """
        if not self.vote_status == VoteStatus.UNVOTED:
            if self.vote >= 0:
                return EdgeDirection.FORWARD
            return EdgeDirection.BACKWARD
        return None

    # NOTE: VOTE
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

    # NOTE: VOTE
    def get_edge_updates(self):
        """ Return vote-preferred edges for corresponding actions.

            Note:
                Skip already elected edge changes.
        """
        if self.vote_status & VoteStatus.ELECTED:
            return dict()
        self.vote_status |= VoteStatus.ELECTED
        if self.vote_status == VoteStatus.UNVOTED:
            return {EdgeActionTyp.ADD: self._edges[self._vote_result()]}
        return self._edge_action

    def get_edges(self, consider_vote=True):
        """ Return available edges under evaluated vote.

            Note:
                If no agent has claimed interest, all edges are returned.
        """
        if consider_vote:
            self.evaluate_vote()
            return self._edges[self._vote_result()].values()
        return self._edge_registry.values()

    # NOTE: VOTE
    def evaluate_vote(self):
        """ Evaluate voting and stage edges for removal and addition.

            Note:
                In order to reset voting and activating all edges use
                _reset_vote().
        """
        direction = self._vote_result()
        direction_reverse = EdgeDirection.reverse(direction)
        if self._previous_direction == direction_reverse:
            self._edge_action[EdgeActionType.ADD] = self._edges[direction]
            self._edge_action[EdgeActionType.REMOVE] = self._edges[direction_reverse]
            self._previous_direction = direction
            return

    # NOTE: VOTE
    def parse_agent_vote(self, state, agent):
        """ Register interest to use an edge in certain direction.

            Note:
                Agents in AgentMode.STALE are casted to
                    agent.mode = AgentMode.EXPLORING 
                once deactivated edge becomes reenabled. (see exit trigger)
        """
        ids = agent.edge_container_ids()
        edge_direction = self.state2direction[state]
        self.vote += edge_direction
        self._agent_registry[edge_direction][agent.id] = None
        self.vote_status = VoteStatus.VOTED

    # NOTE: VOTE
    def _reset_vote(self):
        """ Reset vote and allow all edge directions to be used.

            Note
                Reset_vote only possible through exit trigger
                with active_agents and registered_agents being zero
        """
        self.vote = 0
        self.vote_status = VoteStatus.UNVOTED
        self.previous_direction = None

    # NOTE: VOTE
    def force_vote(self, edge_direction, agent):
        """ Enforce directional reservation for agents starting on edge. """
        self._reset_vote()
        self.vote += edge_direction
        self._agent_registry[edge_direction][agent.id] = None
        self.vote_status = VoteStatus.VOTED

    # NOTE: VOTE
    def unregister_agent(self, agent_id):
        """ Reserve usage of edge_container in future use. 

            Note:
                This is overwritten with 

            Todo:
                #-> Detect blocked agents and remove themselves from
                    all future sections to maybe free up some sections
                    for other agents.
        """
        self.registered_agents.pop(agent_id, None)

    # NOTE: VOTE
    def register_agent(self, agent_id):
        """ Reserve usage of edge_container in future use. 

            Note:
                This is overwritten with 

            Todo:
                #-> Detect blocked agents and remove themselves from
                    all future sections to maybe free up some sections
                    for other agents.

            Todo:
                - VOTE : test if to run exit() routine if all future
                         agents have unregistered and no active agent

        """
        self.registered_agents[agent_id] = None

    # NOTE: VOTE
    def enter(self, state, agent_id):
        """ Conduct entry procedure for agent from state.

            Note:
                The global priority_dict helps in selecting
                promising candidates for the rollout.

            Todo:
                Register agents globally to improve the priority
                dict for rollout.
        """
        self.active_agents[agent_id] = self.agents[agent_id]

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
        if not eta:
            self.exit(agent_id)
        return eta

    # NOTE: VOTE
    def _update_occupancy(self):
        """ Conduct freeing routine when last active agent leaves edge.

            Todo:
                Reverse direction for affected agent retrieval.

        """
        if not len(self.active_agents) and not len(self.registered_agents):
            # NOTE: VOTE - Skip voting and registration
            return
            edge_direction = self._vote_result()
            if edge_direction is not None:
                edge_direction_reverse = EdgeDirection.reverse(edge_direction)
                self._edge_action[EdgeActionType.ADD] = self._edges[edge_direction_reverse]
            else:
                self._edge_action[EdgeActionType.ADD] = self._edge_registry
            self.edge_reactivation[self.id] = self
            self._reset_vote()
            for agent_id in self._agent_registry[edge_direction].copy().keys():
                self.agents[agent_id].update_edge_availability(self.id)
                self._agent_registry[edge_direction].pop(agent_id, None)

    # NOTE: VOTE
    def exit(self, agent_id):
        """ Remove agent from active_agents and priority_dict. """
        # prio = self.active_agents[agent_id].priority
        # self.priority_dict[prio].pop(agent_id, None)
        self.active_agents.pop(agent_id, None)
        self.registered_agents.pop(agent_id, None)
        # NOTE: VOTE - skip self._update_occupancy()

