#!/bin/env python

class Display:

    def report_states(self, states):
        """ Report states via their states_containers and their metrics. """
        for vertex in vertices.keys():
            pass
            #print('\tVertex: \t{}'.format(vertex))
            #print('\tControl: \t{}'.format(self.states[vertex]))

    def show_states(self, env_renderer):
        """ Show states defined in StatesContainer through Flatland utilities. """
        #env_renderer.renderer.plot_single_agent((16,22), 1, 'r', target=(0,0),selected=True)
        #return
        v = list(vertices.keys())[0]
        controls = self.states[v]
        l = list()
        for v in vertices.keys():
            for control in controls:
                l += [Direction2Target[control.direction]]
                c = Transition2Color[control.direction]
                env_renderer.renderer.plot_single_agent((v.r, v.c), v.d, 'r',selected=True)

                if self._show_transitions:
                    env_renderer.renderer.plot_transition(
                            position_row_col=(v.r, v.c),
                            transition_row_col=l,
                            color=c
                            )

