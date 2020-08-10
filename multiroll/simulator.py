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
        self.occupancy = [Occupancy.FREE] * len(self.railway.values())

    def _init_sim_occupancy(self):
        self.occupancy = [Occupancy.FREE] * len(self.railway.values())
        for agent in self.sim_agents.values():
            coc_id = self.states[agent.state].coc.id
            self.occupancy[coc_id] = Occupancy.OCCUPIED

    def _reset_sim_agents(self):
        """ Reset all SimAgentContainer to AgentContainer states. """
        self.occupancy = [Occupancy.FREE] * len(self.railway.values())
        for agent in self.sim_agents.values():
            agent.reset()

    def _transition(self, agent):
        """ Return true for successful transition and update agent state. """
        control = agent.controller[agent.state]
        state_next = Simulator(agent.state, control)
        coc_next = self.states[state_next].coc
        coc_now = self.states[agent.state].coc

        #print(self.occupancy)
        #print('From', agent.state, ' to ', state_next)
        #print('next:' ,self.occupancy[coc_next.id]) 
        #print('now:' ,self.occupancy[coc_now.id]) 
        #print('coc_now.id:',coc_now.id)
        #print('coc_next.id:',coc_next.id)
        if self.occupancy[coc_next.id] == Occupancy.OCCUPIED:
            # print('Transition failed!')
            return False
        # print('Transition succeded!')
        self.occupancy[coc_now.id] = Occupancy.FREE
        self.occupancy[coc_next.id] = Occupancy.OCCUPIED
        agent.state = state_next
        return True

    def _cost_for_commuters(self, agent):
        """ Penalise all agents that are in commute and not arrived yet. """
        target_id = self.railway[agent.target].id
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
            cost += self._cost_for_commuters(agent)
            if not self._transition(agent):
                cost += Cost.NO_TRANSITION
                continue
        return cost

    def simulate_steps(self, steps):
        """ Simulate steps and return total cost. """

        self._init_sim_occupancy()
        cost = 0
        for step in range(steps):
            cost += self._simulate_step()
        return cost


class SimAgentContainer(multiroll.agent.AgentContainer):
    """ Container that collects simulation related elements ."""

    def __init__(self, agent_native):
        self.id = agent_native.id
        self.path = agent_native.path
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
        self.path = self._agent.path.copy()
        self.state = self._agent.state
        self.status = self._agent.status
        self.controller = self._agent.controller.copy()

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

