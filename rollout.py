#!/bin/env python



import graph.py
import simulator.py



class Rollout:
    """ 

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
    def __init__(self):
        # Dictionary of agents with key: agent_id and value: agent_container
        self.agents = None
        # Integer for prediction horizon
        self.prediction_horizon = None
        # priority dictionary: sorted agents according to their current edge
        self.priority_dict = None
        # simulator
        self.simulator = None

    def _rollout(self, agent_id):
        """ Run the rollout for one agent.

            Note:
                Selects the best control based on cost
                returned from simulator.

        """
        ## get controls
        # steps = self.rediction_horizon
        # for each control in controls
        #   agent_control = dict()
        #   agent_control[agent_id] = control
        #   total_cost = self.simulator.simulate_steps(agent_control,
        #                                              steps)

        pass

    def rollout(self):
        """ Execute rollout simulation for each agent. 

            Note:
                Returns the controls dictionary for all agents 
                at current stage.
        """
        # for each agent
        #   self._rollout
        pass


