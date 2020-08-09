#!/bin/env python


from multiroll import *

from .constants import *
from .framework import *


class Rollout(multiroll.simulator.Simulation):
    """ Rollout implementation for networkX graph. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Integer for prediction horizon
        self.prediction_horizon = 30
        # priority dictionary: sorted agents according to their current edge
        self.priority_dict = None
        # Controls indexed by agent_id and action
        self._controls = dict()

        for agent in self.agents.values():
            agent.update()
            agent.locate()
            path, heuristic, status = self.compute_heuristic(agent)

            if status == PathStatus.INFEASIBLE:
                agent.path_status = status
                raise RuntimeError()
                continue
            agent.set_controller(path, heuristic)

    def update_agent_states(self):
        """ Update each agent state with most recent flatland states.

            Note:
                This also updates the agent's current graph node 
                agent.current_node for the following heuristic algorithm.

        """
        for agent in self.agents.values():
            agent.update()

    def _rollout(self):
        """ Execute rollout simulation for each agent.

            Note:
                Returns the controls dictionary for all agents
                at current stage.

            Todo:
                Use jit or keep context local with env update inside
        """
        d_M = self.prediction_horizon
        for agent in self.sim_agents.values():
            # TODO: Test
            if agent.status == AgentStatus.ON_PATH:
                self._controls[agent.id] = agent.controller[agent.state]
                continue
            self._reset_sim_agents()
            state = agent.state

            # Get all available controls at state
            controls = self.states[state].controls

            # Costs indexed by corresponding control
            costs = dict()
            # Heuristics indexed by controls at current step
            heuristics = dict()

            # Get active heuristic cost
            control_heuristic = agent.controller[state]
            cost = self.simulate_steps(d_M)
            costs[control_heuristic] = cost

            # Get cost for remaining control choices
            controls.pop(controls.index(control_heuristic))
            for control in controls:
                path, heuristic, status = self.compute_heuristic(agent)
                if status == PathStatus.INFEASIBLE:
                    cost[control] = Cost.INFEASIBLE
                    print('Graph generation requires serious inspection.')
                    raise RuntimeError()
                agent.heuristic = heuristic
                cost = self.simulate_steps(d_M)
                heuristics[control] = (path, heuristic)
                costs[control] = cost

            min_control = min(costs, key=costs.get)
            if min_control != control_heuristic:
                agent.set_controller(heuristics[min_control])
            self._controls[agent.id] = min_control.control

    def controls(self):
        """ Conduct rollout and return the control. """
        self._rollout()

        return self._controls

