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

            state = agent.state
 
            # Get all available controls at state
            controls = self.states[state].controls.copy()

            # Add stop control

            print('###controls:', controls)
            raise RuntimeError()
            # Costs indexed by corresponding control
            costs = dict()
            # Heuristics indexed by controls at current step
            heuristics = dict()

            # Get active heuristic cost
            sc = self.states[agent.state]
            control_heuristic = agent.controller[agent.state]
            cost = self.simulate_steps(d_M)
            print('All controls:', controls)
            print('State', agent.state)
            #print('Heuristic', agent.controller)
            #for state in agent.controller.keys():
            #    print('State: \t{} \t{} \t{}'.format(state.r, state.c, state.d))
            #print('Edges', sc.edges)
            #print('Agent edges', agent._agent.path)
            #print('Targets', agent.target_nodes)
            print('base cost:', cost)
            print('control:', control_heuristic)
            costs[control_heuristic] = cost
            self._reset_sim_agents()

            # Get cost for remaining control choices
            controls.pop(controls.index(control_heuristic))
            for control in controls:
                path, heuristic, status = self.compute_heuristic(agent)
                if status == PathStatus.INFEASIBLE:
                    costs[control] = Cost.INFEASIBLE
                    print('Graph generation requires serious inspection.')
                    raise RuntimeError()
                agent.heuristic = heuristic
                cost = self.simulate_steps(d_M)
                print('current cost:', cost)
                print('control:', control)
                heuristics[control] = (path, heuristic)
                costs[control] = cost
                self._reset_sim_agents()

            min_control = min(costs, key=costs.get)
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
        self._rollout()

        return self._controls

