import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from pylab import colorbar, cm, imshow
import time

from utils.util_fns import *
from utils.constants import *


class SafetyGrid:
    '''A class that discretizes the workspace into a grid. Each square in the
    grid contains information that can be used during an experiment to help
    implement safety measures related to the ArmAssist xy-position, ArmAssist 
    psi (orientation) angle, and ReHand pronosupination angle.
    '''

    # each square in the grid is of this dtype
    dtype = np.dtype([('is_valid',  np.bool_),
                      ('min_psi',   np.float64),
                      ('max_psi',   np.float64),
                      ('min_prono', np.float64),
                      ('max_prono', np.float64)])

    def __init__(self, mat_size, delta):
        self.mat_size = mat_size  # in cm (e.g., see settings.py)
        self.delta = delta  # size of each square (e.g., width in cm)

        self.grid_shape = (
            np.ceil(mat_size[1] / delta),  # nrows <--> y-size of mat
            np.ceil(mat_size[0] / delta),  # ncols <--> x-size of mat
        )
        self._grid = np.zeros(self.grid_shape, dtype=self.dtype)

        self._grid['min_psi']   = np.nan
        self._grid['max_psi']   = np.nan
        self._grid['min_prono'] = np.nan
        self._grid['max_prono'] = np.nan

    def _pos_to_square(self, pos):
        return (int(pos[1] / self.delta), int(pos[0] / self.delta))

    def _square_to_pos(self, square):
        return (square[1] * self.delta, square[0] * self.delta)

    def _is_square_on_grid(self, square):
        return (0 <= square[0] < self._grid.shape[0]) and \
               (0 <= square[1] < self._grid.shape[1])

    def is_valid_pos(self, pos):
        return self._grid[self._pos_to_square(pos)]['is_valid']

    def get_minmax_psi(self, pos):
        '''Return a tuple with the min and max psi angle for a given xy-position.'''
        
        square = self._pos_to_square(pos)
        return self._grid[square]['min_psi'], self._grid[square]['max_psi']

    def get_minmax_prono(self, pos):
        '''Return a tuple with the min and max prono angle for a given xy-position.'''
        
        square = self._pos_to_square(pos)
        return self._grid[square]['min_prono'], self._grid[square]['max_prono']

    def set_valid_boundary(self, positions):
        '''Given a list of positions defining the outer boundary, mark the
        corresponding squares as valid in the underlying grid representation.
        Positions should have shape (n_positions, 2). The first and last 
        position should be roughly the same.
        '''

        # iterate through each position in the list of positions
        for idx in range(positions.shape[0]):
            pos = positions[idx, :]

            if idx == positions.shape[0] - 1:
                next_pos_idx = 0
            else:
                next_pos_idx = idx + 1
            next_pos = positions[next_pos_idx, :]

            n_pts = max(3, int(10 * (dist(pos, next_pos) / self.delta)))

            # consider equally-spaced pts on a virtual line from pos to next_pos
            for weight in np.linspace(0, 1, n_pts):
                pos_on_line = (1-weight) * pos + weight * next_pos
                square = self._pos_to_square(pos_on_line)

                # for each pt on this line, mark the corresponding square in 
                # the safety grid as a valid position
                self._grid[square]['is_valid'] = True

    def update_minmax_psi(self, pos, psi, local_dist):
        '''Given an xy-position and a psi angle value, update the min/max 
        psi value for all squares within a radius of local_dist cm.
        '''

        square = self._pos_to_square(pos)
        n_squares = int(np.ceil(local_dist / self.delta))

        row_start = max(square[0] - n_squares, 0)
        row_end   = min(square[0] + n_squares, self._grid.shape[0] - 1)
        col_start = max(square[1] - n_squares, 0)
        col_end   = min(square[1] + n_squares, self._grid.shape[1] - 1)

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                square = (row, col)
                if self._grid[square]['is_valid'] and dist(pos, self._square_to_pos(square)) <= local_dist:
                    # min() and max() functions will overwrite np.nan values
                    self._grid[square]['min_psi'] = min(psi, self._grid[square]['min_psi'])
                    self._grid[square]['max_psi'] = max(psi, self._grid[square]['max_psi'])

    def update_minmax_prono(self, pos, prono, local_dist):
        '''Given an xy-position and a prono angle value, update the min/max
        prono value for all squares within a radius of local_dist cm.
        '''

        square = self._pos_to_square(pos)
        n_squares = int(np.ceil(local_dist / self.delta))

        row_start = max(square[0] - n_squares, 0)
        row_end   = min(square[0] + n_squares, self._grid.shape[0] - 1)
        col_start = max(square[1] - n_squares, 0)
        col_end   = min(square[1] + n_squares, self._grid.shape[1] - 1)

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                square = (row, col)
                if self._grid[square]['is_valid'] and dist(pos, self._square_to_pos(square)) <= local_dist:
                    # min() and max() functions will overwrite np.nan values
                    self._grid[square]['min_prono'] = min(prono, self._grid[square]['min_prono'])
                    self._grid[square]['max_prono'] = max(prono, self._grid[square]['max_prono'])

    def mark_interior_as_valid(self, interior_pos):
        '''Assuming that the boundary of valid positions has already been set,
        marks all the squares in the interior as valid too. The argument 
        interior_pos should be a xy-position that is known to be in the 
        interior.
        '''

        # starting at interior_pos, do a breadth-first traversal of squares
        #   (using a queue) and use a set to keep track of which squares have
        #   already been visited
        starting_square = self._pos_to_square(interior_pos)
        queue = deque([starting_square])
        visited = set()

        while len(queue) > 0:  # while the queue is not empty
            # get the next square to be processed (i.e., to be marked as valid)
            square = queue.popleft()

            self._grid[square]['is_valid'] = True
            visited.add(square)

            row = square[0]
            col = square[1]
            # if a neighboring square is:
            #   1) on the grid,
            #   2) hasn't already been visited,
            #   3) isn't already in the queue, and
            #   4) isn't already marked as valid
            # then mark add it to the queue 
            for neighbor in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                if neighbor not in visited and neighbor not in queue:
                    if self._is_square_on_grid(neighbor):
                        if not self._grid[neighbor]['is_valid']:
                            queue.append(neighbor)

    def is_psi_minmax_set(self):
        '''Return true if the min/max psi value is set for all squares that 
        are marked as valid.
        '''

        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row][col]['is_valid'] and np.isnan(self._grid[row][col]['min_psi']):
                        return False
        return True

    def is_prono_minmax_set(self):
        '''Return true if the min/max prono value is set for all squares that 
        are marked as valid.
        '''

        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row][col]['is_valid']:
                    if np.isnan(self._grid[row][col]['min_prono']) or np.isnan(self._grid[row][col]['max_prono']):
                        return False
        return True

    def plot_valid_area(self):
        '''Plot the valid xy area of the workspace.'''

        extent = [0, self.mat_size[0], 0, self.mat_size[1]]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self._grid[:]['is_valid'], 
            interpolation='none', extent=extent, origin='lower')
        ax.set_title('Valid ArmAssist positions')
        ax.set_xlabel('cm')
        ax.set_ylabel('cm')

    def plot_minmax_psi(self):
        '''Plot the min/max psi for each position.'''

        extent = [0, self.mat_size[0], 0, self.mat_size[1]]

        global_min = rad_to_deg * np.nanmin(self._grid[:]['min_psi'])
        global_max = rad_to_deg * np.nanmax(self._grid[:]['max_psi'])

        fig, axes = plt.subplots(nrows=1, ncols=2)
        variables = ['min_psi', 'max_psi']
        titles = ['Min psi angle', 'Max psi angle']
        for ax, var, title in zip(axes.flat, variables, titles):
            matrix = rad_to_deg * self._grid[:][var]
            for row in range(self._grid.shape[0]):
                for col in range(self._grid.shape[1]):
                    if self._grid[row][col]['is_valid'] and np.isnan(self._grid[row][col][var]):
                        matrix[row][col] = 1e6

            im = ax.imshow(matrix, 
                interpolation='none', origin='lower', extent=extent, vmin=global_min, vmax=global_max)
            ax.set_title(title + '\n(black = no value)')
            ax.set_xlabel('cm')
            ax.set_ylabel('cm')
        
        im.cmap.set_over('k')
        im.set_clim(global_min, global_max)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_title('degrees')
        fig.colorbar(im, cax=cbar_ax)

    def plot_minmax_prono(self):
        '''Plot the min/max prono angle for each position.'''

        extent = [0, self.mat_size[0], 0, self.mat_size[1]]

        global_min = rad_to_deg * np.nanmin(self._grid[:]['min_prono'])
        global_max = rad_to_deg * np.nanmax(self._grid[:]['max_prono'])
        if np.isnan(global_min):
            global_min = 0
            global_max = 1

        fig, axes = plt.subplots(nrows=1, ncols=2)
        variables = ['min_prono', 'max_prono']
        titles = ['Min prono angle', 'Max prono angle']
        for ax, var, title in zip(axes.flat, variables, titles):
            matrix = rad_to_deg * self._grid[:][var]
            for row in range(self._grid.shape[0]):
                for col in range(self._grid.shape[1]):
                    if self._grid[row][col]['is_valid'] and np.isnan(self._grid[row][col][var]):
                        matrix[row][col] = 1e6

            im = ax.imshow(matrix, 
                interpolation='none', origin='lower', extent=extent, vmin=global_min, vmax=global_max)
            ax.set_title(title + '\n(black = no value)')
            ax.set_xlabel('cm')
            ax.set_ylabel('cm')

        im.cmap.set_over('k')
        im.set_clim(global_min, global_max)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_ax.set_title('degrees')
        fig.colorbar(im, cax=cbar_ax)

    def calculate_valid_area(self):
        '''Return estimate of the range of motion area in cm^2.'''
        
        n_valid_squares = 0
        for row in range(self._grid.shape[0]):
            for col in range(self._grid.shape[1]):
                if self._grid[row][col]['is_valid']:
                    n_valid_squares += 1

        return n_valid_squares * self.delta**2


if __name__ == '__main__':
    # below is a simple example of how a SafetyGrid can be created
    # see the scripts:
    #   define_safety_boundary.py
    #   define psi_prono_safety_range.py
    # for full usage

    safety_grid = SafetyGrid([95, 85], 0.5)

    interior_pos = (50, 50)
    angles = np.linspace(0, 2 * np.pi, 100)
    radius = 10
    x = radius*np.cos(angles) + 0.1*np.random.randn(len(angles)) + interior_pos[0]
    y = radius*np.sin(angles) + 0.1*np.random.randn(len(angles)) + interior_pos[1]
    boundary_positions = np.array([x, y]).T

    safety_grid.set_valid_boundary(boundary_positions)
    safety_grid.mark_interior_as_valid(interior_pos)
    safety_grid.plot_valid_area()

    plt.show()
