#!/bin/env python

import graph


class SimAgentContainer(graph.AgentContainer):
    """ 
        Note:
            agent_container:
                - add path as state to control dictionary
                - simulator -> state to control heuristic
    """
    def __init__(self, agent_container):
        self.agent = agent_container

        # Initialise default heuristic
        self.heuristic = self.agent.heuristic.copy()

    def update_heuristic(self):
        self.agent.heuristic = self.heuristic


class Occupancy:
    FREE = 0
    OCCUPIED = 1


class Cost:
    NONE = 0
    NO_TRANSITION = 100
    NOT_AT_TARGET = 10


class Simulator:
    """
        # SIMULATOR
        # mirror current agents and their states
        # keep track of agent steps and returns control based on agent states at each step
        # -> agent_id -> sim_container -> agent_edge_cell ->  control
        # -> simulator returns if agent transitions successfully or not
        Note:
            agent_container -> attribute sim_state
            agent_container -> method: sim_controls
            agent_container -> method: sim_reset: reset state to actual state and sim_heuristic
            agent_container -> attribute heuristic: dict[cell_id_of_path] = control
            agent_container -> attribute sim_heuristic
    """
    def __init__(self, env):
        # Agents dictionary with agent_id -> agent_container (global)
        self.agents = None
        # Graph reference -> provides shortest path heuristic
        graph.set_env(env)
        self.graph = graph.MyGraph() #TODO: initialise in constructor
        self.rail_env_mirror = FlatlandMirror(env) #TODO: copy all environment elements
        # define occupancy 
        pass

    def _initialise(self):
        # TODO: get each AgentContainer and create a SimAgentContainer
        pass

    def _reset_agents(self):
        # TODO: go through all SimAgentContainers is self.agents
        pass

    def update_heuristic(self, agent_id, control):
        """ Update heuristic for agent. """
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
        for step in range(steps)
            cost += self._simulate_step()
        return cost

