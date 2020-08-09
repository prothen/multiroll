#!/bin/env python


from multiroll import *

from .constants import *
from .framework import *


class Rollout(multiroll.simulator.Simulator):
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
    def __init__(self, *args, **kwargs):
        """ 

            Note:
                Specify desired keywords for this class explicitly
        """
        super().__init__(*args, **kwargs)
        # Integer for prediction horizon
        self.prediction_horizon = 30
        # priority dictionary: sorted agents according to their current edge
        self.priority_dict = None
        # Controls indexed by agent_id and action
        self._controls = dict()

    def _rollout(self):
        """ Execute rollout simulation for each agent. 

            Note:
                Returns the controls dictionary for all agents 
                at current stage.

            Todo:
                Consider edge priority and agent.state.status

            Todo:
                Use jit or keep context local with env update inside
        """
        d_M = self.prediction_horizon
        self.sim_agents.reset()
        for agent_id in self.agents.keys():
            self._rollout(agent_id)
            agent = self.sim_agents[agent_id]
            state = agent.state

            # Get all available controls at state
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
                heuristics[control] = agents[agent_id].heuristic
                costs[control] = cost

            min_control = min(costs, key=costs.get)
            if min_control != control_heuristic:
                # sim agents know which control has which heuristic
                agent.update_heuristic(heuristics[min_control])
            self._controls[agent.id] = min_control.control

    def controls(self):
        """ Conduct rollout and return the control.

            Note:
                Agents are transitioned in rollout()

        """
        self._rollout()

        return self._controls

