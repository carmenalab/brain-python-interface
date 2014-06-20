'''
Classes to apply hard (nonlinear) limits to the state of the Decoder.
Example: screen limits on a cursor position to keep it always in view. 
'''
import numpy as np

class RectangularBounder(object):
    ''' Docstring '''
    def __init__(self, min_vals, max_vals, states_to_bound):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.bounding_box = (np.array(min_vals), np.array(max_vals))
        self.states_to_bound = states_to_bound

    def __call__(self, state, state_ls):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if not self.bounding_box == None:
            state_inds = np.array([state_ls.index(x) for x in self.states_to_bound])

            min_bounds, max_bounds = self.bounding_box
            repl_with_min = np.asarray(state)[state_inds, 0] < min_bounds
            state[state_inds[repl_with_min], 0] = min_bounds[repl_with_min]

            repl_with_max = np.asarray(state)[state_inds, 0] > max_bounds
            state[state_inds[repl_with_max], 0] = max_bounds[repl_with_max]
        return state


def make_rect_bounder_from_ssm(ssm):
    min_vals = []
    max_vals = []
    states_to_bound = []
    for state in ssm.states:
        if not ((state.min_val == np.nan) or (state.max_val == np.nan)):
            min_vals.append(state.min_val)
            max_vals.append(state.max_val)
            states_to_bound.append(state.name)

    return RectangularBounder(min_vals, max_vals, states_to_bound)
