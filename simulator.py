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
    def __init__(self):
        # Agents dictionary with agent_id -> agent_container (global)
        self.agents = None
        # Graph reference -> provides shortest path heuristic
        self.graph = None
        # Initialise graph
        self.graph.initialise()
        pass

    # SIMULATOR method
    def _get_agents_control(self):
        """ EXAMPLE docstring:
                Return all agent heuristics
        """
        agent_controls = dict()
        for agent in self.agents:
            agent_controls[agent.id] = agent.sim_control
        return agent_controls

    def simulate_step(self, controls):
        # for each agent:
            # update states
            # check if cells are free
            # update agents sim_state
        # return cost
        pass

    # SIMULATOR method
    def simulate_steps(self, agent_control, steps):
        """
           
           # TO BE DONE in rollout.py
           agent_container:
                    update heuristic for rollout agent
                    store heuristics temporary (edge_id list)
        """
        # RESET simulation
        # reset all agents
        ## agent_container.reset_sim()

        ## 
        ## simulate one step
        ## rail_env follows agent_ids (sequential)

        ## control = agent_control.value()
        ## if control not agents[agent_id].heuristic[0]:
        ##      recompute heuristic (graph)
        ##

        ## simulate agents
        ## compute heuristic for agent_id
        ## find current edge (state, control) pair
        ## update path for current agent_id

        ## get agents control
        # agent_controls = self.simulator_get_agents_control(0)
        ## overwrite agent_id heuristic
        # agent_controls.update(**agent_control)
