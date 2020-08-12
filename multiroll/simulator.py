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

        self.sim_agents_active = dict()
        self.sim_agents_active.update(self.sim_agents)

        # Define state.id sorted occupancy of all states at current sim step
        self.occupancy = [Occupancy.FREE] * len(self.railway.values())

    def _update_active_agents(self):
        agents_to_drop = list()
        for agent in self.sim_agents_active.values():
            if agent.status == AgentStatus.ON_TARGET:
                agents_to_drop.append(agent.id)
                continue
        for agent_id in agents_to_drop:
            self.sim_agents_active.pop(agent_id, None)

    def _reset_sim(self):
        """ Reset all simulation related objects and initialise correspondingly. """
        for agent in self.sim_agents_active.values():
            agent.reset()

        self.occupancy = [Occupancy.FREE] * len(self.railway.values())
        for agent in self.sim_agents_active.values():
            coc_id = self.states[agent.state].coc.id
            self.occupancy[coc_id] = Occupancy.OCCUPIED

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

    def simulate_steps(self, steps):
        """ Simulate steps and return total cost. """

        print('Simulate for: ', len(self.sim_agents_active))
        cost = 0
        for step in range(steps):
            for agent in self.sim_agents_active.values():
                target_id = self.railway[agent.target].id

                if target_id == self.states[agent.state].coc.id:
                    self.status = AgentStatus.ON_TARGET
                    coc_now = self.states[agent.state].coc
                    self.occupancy[coc_now.id] = Occupancy.FREE
                    continue
                cost += Cost.NOT_AT_TARGET

                control = agent.controller[agent.state]
                state_next = Simulator(agent.state, control)
                coc_next = self.states[state_next].coc
                coc_now = self.states[agent.state].coc

                if True:
                    #print(self.occupancy)
                    print('\nAgent: ', agent.id)
                    print('From', agent.state, ' to ', state_next)
                    print('Control: ', control)
                    print('Occupancy:' , self.occupancy[coc_now.id],
                          '->' ,self.occupancy[coc_next.id]) 
                if Occupancy(self.occupancy[coc_next.id]) & Occupancy.OCCUPIED:
                    if control == DontMoveControl:
                        cost += Cost.NOT_AT_TARGET
                        continue
                    cost += Cost.NO_TRANSITION
                    continue
                self.occupancy[coc_now.id] = Occupancy.FREE
                self.occupancy[coc_next.id] = Occupancy.OCCUPIED
                agent.state = state_next
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
        # TODO: REMOVE duplicate
        self.path = path
        self.controller = controller.copy()
        self._agent.set_controller(path, controller)

    def get_control(self):
        """ Return control from underlying AgentContainer.

            Note:
                Meant for deployed real control usage.
                Not for SimAgentContainer related actions.

        """
        return self._agent.controller[self.state].control
 
    def update(self):
        # TODO: debug update
        self._agent.update()
