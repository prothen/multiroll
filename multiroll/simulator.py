#!/bin/env python


from multiroll import *

from .constants import *


class Simulation(multiroll.heuristic.ShortestPath):
    """ Simulator for cost computation along N steps. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sim_agents = dict()
        for agent in self.agents.values():
            self.sim_agents[agent.id] = SimAgentContainer(agent)

        # Define state.id sorted occupancy of all states at current sim step
        self.occupancy = [0] * len(self.railway.values())

    def _reset_sim_agents(self):
        """ Reset all SimAgentContainer to AgentContainer states. """
        for agent in self.sim_agents.values():
            agent.reset()

    def _transition(self, agent):
        """ Return true for successful transition and update agent state. """
        control = agent.controller[agent.state]
        state_next = Simulator(agent.state, control)
        coc_next = self.states[state_next]

        if self.occupancy[coc_next.id] == Occupancy.OCCUPIED:
            return False

        coc_now = self.states[agent.state]
        self.occupancy[coc_now.id] = Occupancy.FREE
        self.occupancy[coc_next.id] = Occupancy.OCCUPIED
        agent.state = state_next
        return True

    def _cost_for_commuters(self, agent):
        """ Penalise all agents that are in commute and not arrived yet. """
        target_id = self.railway[agent.target]

        if not target_id == self.states[agent.state].coc.id:
            return Cost.NOT_AT_TARGET
        return Cost.NONE

    def _simulate_step(self):
        """ Return cost for all SimAgentContainers according to their
            recent heuristic moving one step.

        """
        cost = 0
        # TODO: agents currently unordered! use collections.OrderedDict()
        for agent in self.sim_agents.values():
            if not self._transition(agent):
                # print('occupied or failure to transition')
                cost += Cost.NO_TRANSITION
                continue
            cost += self._cost_for_commuters(agent)
        return cost

    def simulate_steps(self, steps):
        """ Simulate steps and return total cost. """
        self._reset_sim_agents()

        cost = 0
        for step in range(steps):
            cost += self._simulate_step()
        return cost


class SimAgentContainer(multiroll.agent.AgentContainer):
    """ Container that collects simulation related elements ."""

    def __init__(self, agent_native):
        self.id = agent_native.id
        self.state = agent_native.state
        self.status = agent_native.status
        # target coordinate
        self.target = agent_native.target
        # target states
        self.target_nodes = agent_native.target_nodes

        # Define current simulation heuristic
        self.controller = agent_native.controller.copy()

        # Store a reference to its native base AgentContainer
        self._agent = agent_native

    def reset(self):
        """ Reset agent state to its AgentContainer base. """
        self.state = self._agent.state
        self.status = self._agent.status
        self.controller = self._agent.controller

    def set_controller(self, path, controller):
        """ Update native AgentContainer with new controller. """
        self._agent.set_controller(path, controller)

    def get_control(self):
        """ Return control from underlying AgentContainer.

            Note:
                Meant for deployed real control usage.
                Not for SimAgentContainer related actions.

        """
        return self._agent.controller[self.state].control

