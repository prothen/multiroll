#!/bin/env python


from constants import *
from framework import *

import simulator


class Rollout:
    """ 
        
        Rollout only interacts with simulator to get flatland metrics. So
        only simulator receives env argument.
    
        Note:
            agent_container:
                - should provide feed_forward control to simulator
                - should have current state
                - should show if it has control choices at current step

            simulator:
                - move agents along prediction_horizon (track their states)
                - compute reward for each prediction step
                - provide method to return cost given agent_states


    """
    def __init__(self, env):
        self.graph = graph.MyGraph(env)
        # Dictionary of agents with key: agent_id and value: agent_container
        # TODO: initialise agents from simulator
        self.agents = None
        # Integer for prediction horizon
        self.prediction_horizon = None
        # priority dictionary: sorted agents according to their current edge
        self.priority_dict = None
        # simulator
        self.simulator = simulatorNone
        # Controls indexed by agent_id and action
        self.controls = dict()


    def _rollout(self, agent_id):
        """ Run the rollout for one agent.

            Note:
                Selects the best control based on cost
                returned from simulator.

                agent_container:
                    - find controls other than heuristic
                    - store heuristic dict (states -> control)
                    - heuristic from all edges -> state to control pair

                simulator:
                    simulate_steps
                    simulate_heuristic_steps

        """
        d_M = self.prediction_horizon
        agent = self.agents[agent_id].state
        state = agent.state
        controls = self.states[state].controls

        # Costs indexed by corresponding control
        costs = dict()
        # Heuristics indexed by controls
        heuristics = dict()
        control_heuristic = agent.heuristic[state]
        cost = self.simulator.simulate_steps(d_M)
        costs[control_heuristic] = cost

        # Remove heuristic
        controls.pop(controls.index(control_heuristic))
        for control in controls:
            if simulator.update_heuristic(agent_id, control):
                cost = simulator.simulate_steps(d_M)
            heuristics[control] = self.agents[agent_id].heuristic
            costs[control] = cost

        min_control = min(costs, key=costs.get)
        if min_control != control_heuristic:
            agent.update_heuristic(heuristics[min_control])
            self.controls[agent.id] = min_control.control

    def rollout(self):
        """ Execute rollout simulation for each agent. 

            Note:
                Returns the controls dictionary for all agents 
                at current stage.
        """
        for agent in self.agents:
            if self.states[agent.state].priority != Priority.None:
                self._rollout(agent)
            else:
                self.controls[agent.id] = agent.heuristic[agent.state].control
        return self.controls


