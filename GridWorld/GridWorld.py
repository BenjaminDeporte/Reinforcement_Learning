# basic GridWorld home-made example
# based on Sutton and Barto's "Reinforcement Learning" book
# example p.77

# --- infos ----------------------------------------------------------------------------------

# the GridWorld is a 2D world of NX x NY squares.

# The goal is to go from any square in the world to the upper-left or lower-right corner of the world, as fast as possible

# A state is the position of a square, with coordinates (x,y) (0<= x <=NX-1, 0 <= y <= NY-1). (0,0) is the upper left corner.

# An action is one of the following four : up, down, right or left. A state does not change if an action attempts to go outside the grid (for example "up" when y=0)

# The MDP dynamcis are DETERMINISTIC in this example : an action will move a state to another state (possibly the same) with probability one.abs

# A policy $\pi$ associates each state with four probabilities (summing up to one) of going up,down,right, left.

# --- librairies -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# --- hyperparameters ------------------------------------------------------------------------

# world size
NX = 4
NY = 4
# discount
GAMMA = 1.0

# --- basic classes --------------------------------------------------------------------------

# - Value Function -------

class ValueFunction():
    """Value function class. Stores value functions for each state, provides basic get, update and display methods
    """
    
    def __init__(self, nx=NX, ny=NY, value_function=None):
        self.nx = nx
        self.ny = ny
        if value_function is None:
            self.vf = np.zeros(shape=(self.nx, self.ny))
        else:
            self.vf = value_function.vf
            
    def get(self, x,y):
        assert (0 <= x < self.nx) and (0 <= y < self.ny), "erreur : hors grid dans ValueFunction.get()"
        return self.vf[x,y]
    
    def update(self, x,y, value):
        assert (0 <= x < self.nx) and (0 <= y < self.ny), "erreur : hors grid dans ValueFunction.update()"
        self.vf[x,y] = value
        
    def display(self):
        print(self.vf)

    def __repr__(self):
        msg = f"Objet ValueFunction taille {self.nx} x {self.ny}"
        return msg
        
    def __str__(self):
        msg = f"Objet ValueFunction taille {self.nx} x {self.ny}"
        return msg       