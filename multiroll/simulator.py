#!/bin/env python


from multiroll import *


class Simulator(multiroll.heuristic.ShortestPath):


    """
        # SIMULATOR
        # mirror current agents and their states
        # keep track of agent steps and returns control based on agent states at each step
        # -> agent_id -> sim_container -> agent_edge_cell ->  control
        # -> simulator returns if agent transitions successfully or not
        Note:
            agent_container -> method: sim_controls
            agent_container -> method: sim_reset: reset state to actual state and sim_heuristic
            agent_container -> attribute heuristic: dict[cell_id_of_path] = control
            agent_container -> attribute sim_heuristic
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_agents = dict()
        for agent in self.agents.values():
            self.sim_agents[agent.id] = SimAgentContainer(agent)

        # Define state.id sorted occupancy for current sim step all states
        self.occupancy = [0] * len(self.states)

    def _reset_sim_agents(self):
        """ Reset all SimAgentContainer to AgentContainer states. """
        for agent in self.sim_agents.values():
            agent.reset()

    def update_heuristic(self, agent_id, control):
        """ Update heuristic for agent. """
        self.update_shortest_path(agent)
        # TODO: Update SimAgentContainer heuristic attribute
        # Get Edge for (state,control) pair control.direction
        # if edge found successful and not blocked by scheduling(TBD -> avoid deadlock)
        #   get next vertex, compute shortest path and update heuristic dictionary 
        #   return True
        return False

    def _transition(self, agent):
        """ Return true for successful transition and update agent state. """
        state_next = graph.Dynamics(agent.state)
        coc_next = self.graph.states[state_next]
        if self.occupancy[coc_next.id] == Occupancy.OCCUPIED:
            return False
        coc_now = self.graph.states[agent.state]
        self.occupancy[coc_now.id] = Occupancy.FREE
        self.occupancy[coc_next.id] = Occupancy.OCCUPIED
        agent.state = state_next
        return True

    def _cost_for_commuters(self, agent):
        """ Penalise all agents that are in commute and not arrived yet. """
        if not agent.target.coc.id == agent.get_coc().id:
            return Cost.NOT_AT_TARGET
        return Cost.NONE

    def _simulate_step(self):
        """ Return cost for all SimAgentContainers according to their
            recent heuristic moving one step.

        """
        cost = 0
        # TODO: agents currently unordered! use collections.OrderedDict()
        for agent in self.agents:
            if not self._transition(agent):
                cost += Cost.NO_TRANSITION
            cost += self._cost_for_commuters(agent)
        return cost

    def simulate_steps(self, steps):
        """

           # TO BE DONE in rollout.py
           agent_container:
                    update heuristic for rollout agent
                    store heuristics temporary (edge_id list)
        """
        self._reset_agents()

        cost = 0
        for step in range(steps):
            cost += self._simulate_step()
        return cost


class Occupancy:
    FREE = 0
    OCCUPIED = 1


class Cost:
    NONE = 0
    NO_TRANSITION = 100
    NOT_AT_TARGET = 10
    INFEASIBLE = 10000000000000


class SimAgentContainer:
    """
        Note:
            agent_container:
                - add path as state to control dictionary
                - simulator -> state to control heuristic
        Todo:
            Not necessary to subclass from AgentContainer
    """
    def __init__(self, agent_native):
        # Store a reference to its native base AgentContainer
        self._agent = agent_native

        self.state = self._agent.state

        # Define current simulation heuristic
        self.heuristic = self._agent.heuristic.copy()

    def reset(self):
        """ Reset agent state to its AgentContainer base. """
        self.state = self._agent.state
        self.heuristic = self._agent.heuristic

    def set_base_heuristic(self, path, heuristic):
        """ Update native AgentContainer with new heuristic. """
        self._agent.set_heuristic(path, heuristic)

