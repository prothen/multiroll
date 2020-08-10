#!/bin/env python


from multiroll import *

from .constants import *
from .framework import *


class Rollout(multiroll.simulator.Simulation):
    """ Rollout implementation for networkX graph. """

    def __init__(self, rollit=False, prediction_steps=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollit = rollit
        # Integer for prediction horizon
        self.prediction_horizon = prediction_steps
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
        import flatland
        d_M = self.prediction_horizon
        for agent in self.sim_agents.values():
            self._reset_sim_agents()
            dest_reached = agent._agent._agent.status == flatland.envs.agent_utils.RailAgentStatus.DONE_REMOVED
            not_vertex =  not self.states[agent.state].type == StateType.VERTEX
            check =  [not_vertex, dest_reached]
            if self.debug_is_enabled:
                print('Agent:', agent._agent._agent.status)
                print('Path:', agent._agent.path_status)
                print('StateType:', self.states[agent.state].type)
                print('Skip rollout checks [not_vertex, dest_reached]')
                print(check)
            if any(check):
                print('Rollout SKIPPED.')
                self._controls[agent.id] = agent.get_control()
                continue
            print('Rollout TRIGGERED.')

            # Get all available controls at state
            state = agent.state
            controls = self.states[state].controls.copy()
            print('###### All controls:', controls)

            # Costs indexed by corresponding control
            costs = dict()
            # Heuristics indexed by controls at current step
            heuristics = dict()

            # Simulate with default heuristic
            control_heuristic = agent.controller[agent.state]
            cost = self.simulate_steps(d_M)
            print('Rollout result for: ', control_heuristic)
            print('     -> Has cost:', cost)
            costs[control_heuristic] = cost
            self._reset_sim_agents()

            # Simulate with Control.S
            stop_control = ControlDirection(Control.S, None)
            agent.heuristic = dict([(state, stop_control)])
            control = agent.controller[agent.state]
            cost = self.simulate_steps(d_M)
            heuristics[stop_control] = (agent._agent.path[0], control)
            costs[stop_control] = cost
            self._reset_sim_agents()
            print('Rollout result for: ', stop_control)
            print('     -> Has cost:', cost)

            # Get cost for remaining control choices
            controls.pop(controls.index(control_heuristic))
            for control in controls:
                path, heuristic, status = self.compute_heuristic(agent)
                if status == PathStatus.INFEASIBLE:
                    costs[control] = Cost.INFEASIBLE
                    print('Graph generation requires serious inspection.')
                    raise RuntimeError()
                    continue
                agent.heuristic = heuristic
                cost = self.simulate_steps(d_M)
                heuristics[control] = (path, heuristic)
                costs[control] = cost
                self._reset_sim_agents()
                print('Rollout result for: ', control)
                print('     -> Has cost:', cost)

            print('costs', costs)

            min_control = min(costs, key=costs.get)
            print('Selected: ', min_control)
            #raise RuntimeError()
            if min_control != control_heuristic:
                print('\n\n#######################')
                print('select new Control :', min_control)
                print('cost:', costs[min_control])
                raise RuntimeError()
                agent.set_controller(heuristics[min_control])
            # self._controls[agent.id] = min_control.control
            self._controls[agent.id] = agent.get_control()

    def controls(self):
        """ Conduct rollout and return the control. """
        if self._rollit:
            self._rollout()
            return self._controls
        return dict([(agent.id, agent.get_control()) for agent in self.agents.values()])

