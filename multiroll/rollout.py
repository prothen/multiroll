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
        d_M = self.prediction_horizon
        self._update_active_agents()
        for agent in self.sim_agents_active.values():
            print('\tAgent:', agent.id)
            print('\tStatus:', agent.status)
            self._reset_sim()
            dest_reached = agent._agent._agent.status == FlatlandAgentStatus.DONE_REMOVED
            not_vertex =  not self.states[agent.state].type == StateType.VERTEX
            check =  [not_vertex, dest_reached]
            if any(check):
                #print('Rollout SKIPPED.')
                self._controls[agent.id] = agent.get_control()
                continue
            print('#########################')
            print('Rollout-Algorithm:')
            if True: # self.debug_is_enabled:
                print('\tAgent:\t', agent.id)
                print('\tStatus:\t', agent.status)
                print('\tStatus (Fl):\t', agent._agent._agent.status)
                print('\tState:\t', agent.state)
                print('\tTarget:\t', agent.target)
                print('\tPath:\t', agent._agent.path_status)
                print('\tStateType:\t', self.states[agent.state].type)
                print('\tSkip rollout checks [not_vertex, dest_reached]')
                print('\t\t', check)

            # Get all available controls at state
            controls = self.states[agent.state].controls.copy()
            print('###### All controls:', controls)

            # Costs indexed by corresponding control
            costs = dict()
            # Heuristics indexed by controls at current step
            heuristics = dict()

            # Simulate with default heuristic
            control_heuristic = agent.controller[agent.state]
            print('Base heuristic: \t', control_heuristic.control)
            cost = self.simulate_steps(d_M)
            print('     -> Has cost:', cost)
            costs[control_heuristic] = cost
            self._reset_sim()

            # Simulate with Control.S
            stop_control = DontMoveControl
            stop_path = [agent.state]
            stop_heuristic = dict([(agent.state, stop_control)])
            controls.append(stop_control)

            # Get cost for remaining control choices
            controls.pop(controls.index(control_heuristic))
            for control in controls:
                print('Alternative controls: \t', control.control)
                if control == stop_control:
                    path = [agent.state]
                    heuristic = dict([(agent.state, stop_control)])
                    status = PathStatus.FEASIBLE
                else:
                    next_edge = self.edge_collection[StateControl(agent.state, control)]
                    edge_container = self.edges[next_edge.container_id]
                    source = edge_container.goal_state[edge_container.state2direction[agent.state]]
                    next_heuristic = next_edge.path[:-1]  # TODO: remove goal state
                    path, heuristic, status = self.compute_heuristic(agent, source)
                    # TODO: this should be unneccessary now heuristic[agent.state] = control
                    heuristic.update(next_heuristic)
                if status == PathStatus.INFEASIBLE:
                    costs[control] = Cost.INFEASIBLE
                    print('Graph generation requires serious inspection.')
                    raise RuntimeError()
                    continue
                agent.controller = (heuristic.copy())
                cost = self.simulate_steps(d_M)
                heuristics[control] = (path, heuristic)
                costs[control] = cost
                self._reset_sim()
                print('     -> Has cost:', cost)

            print('costs', costs)

            min_control = min(costs, key=costs.get)
            print('Selected: ', min_control)
            if min_control != control_heuristic:
                print('\n\n#######################')
                print('select new Control :', min_control)
                print('cost:', costs[min_control])
                agent.set_controller(*heuristics[min_control])
            self._controls[agent.id] = agent.get_control()

    def controls(self):
        """ Conduct rollout and return the control. """
        if self._rollit:
            self._rollout()
            return self._controls
        return dict([(agent.id, agent.get_control()) for agent in self.agents.values()])

