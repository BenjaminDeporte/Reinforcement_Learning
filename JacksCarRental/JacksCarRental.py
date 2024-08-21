#------------------------------------------------------------------
#--- Jack's Car Rental --------------------------------------------
#------------------------------------------------------------------

#--- toy case Sutton & Barto ch4 p81 ------------------------------

#--- librairies ---------------------------------------------------

import numpy as np
import math

#--- parameters ---------------------------------------------------

N_CARS = 20  # maximum number of cars at each rental location
N_TRANSFERTS = 5  # maximum number of cars that can be moved overnight
GAMMA = 0.9 # discount
LAMBDA_CUSTOMERS_1 = 3  # Poisson law parameter for customer requests at location 1
LAMBDA_CUSTOMERS_2 = 4  # Poisson law parameter for customer requests at location 2
LAMBDA_RETURNS_1 = 3 # Poisson law parameter for cars returns at location 1
LAMBDA_RETURNS_2 = 2 # Poisson law parameter for cars returns at location 1

#--- base utilities -----------------------------------------------

#--- Poisson laws : calculate log probas from 0 to N_CARS included -------
#--- NB : probas for n > N_CARS are assumed negligible -------------------

customers_1 = np.array([ n*np.log(LAMBDA_CUSTOMERS_1) - np.log(math.factorial(n))-LAMBDA_CUSTOMERS_1 for n in range(N_CARS+1)])
customers_2 = np.array([ n*np.log(LAMBDA_CUSTOMERS_2) - np.log(math.factorial(n))-LAMBDA_CUSTOMERS_2 for n in range(N_CARS+1)])
returns_1 = np.array([ n*np.log(LAMBDA_RETURNS_1) - np.log(math.factorial(n))-LAMBDA_RETURNS_1 for n in range(N_CARS+1)])
returns_2 = np.array([ n*np.log(LAMBDA_RETURNS_2) - np.log(math.factorial(n))-LAMBDA_RETURNS_2 for n in range(N_CARS+1)])

#-------------------------------------------------------------------------------
#--- base classes --------------------------------------------------------------
#-------------------------------------------------------------------------------

#--- here, the choice is not to use Gym, for learning purposes -----------------

#-------------------------------------------------------------------------------
#--- class for state space -----------------------------------------------------
#-------------------------------------------------------------------------------

class StatesSpace():
    """Encapsulates states space as a N_CARS x N_CARS array, plus some display methods
    Constructor requires n_cars maximum per location
    """
    
    def __init__(self, n_cars=None):
        """Create a StateSpace object, basically a np.array(n_cars x n_cars)

        Args:
            n_cars (_type_, optional): maximum number of cars per agency. Defaults to None.
        """
        if n_cars is None:
            self._n_cars = N_CARS
        else:
            self._n_cars = n_cars
            
        self._states = np.zeros((self.n_cars, self.n_cars))
        
    @property
    def n_cars(self):
        return self._n_cars
    
    @n_cars.setter
    def n_cars(self, n):
        self._n_cars = n
    
    @property
    def states(self):
        return self._states
    
    @states.setter
    def states(self, array_of_states):
        self._states = array_of_states
        
    def display(self):
        print (self.states)
        
    def __repr__(self):
        return f"StateSpace object, size {self.n_cars} x {self.n_cars}"
    
    def __str__(self):
        return f"StateSpace object, size {self.n_cars} x {self.n_cars}"
    
#--- unitary test for StateSpace -----

# states_space = StatesSpace(n_cars = 5)
# print(states_space.states)
# states_space.n_cars = 8
# x = np.ones((states_space.n_cars, states_space.n_cars))
# states_space.states = x
# states_space.display()

#--------------------------------------------------------------------------------
#--- class for action space -----------------------------------------------------
#--------------------------------------------------------------------------------

class ActionsSpace():
    """Ecapsulates data and methods regarding actions.
    Constructor requires n_transferts, which is the maximum number of cars that can be moved overnight.
    By defintion, the number of cars moved from location 1 to location 2 is counted positive.
    """
    
    def __init__(self, n_transferts=None):
        """Create an action space object, basically a np.array(2*n_transferts+1)

        Args:
            n_transferts (_type_, optional): maximum number of cars that can be moved overnight. Defaults to None.
        """
        if n_transferts == None:
            self._n_transferts = N_TRANSFERTS  # default value
        else:
            self._n_transferts = int(n_transferts)
            
        self._actions = np.array( [ n for n in range(-self._n_transferts, self._n_transferts+1)] )
        
    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, x):
        self._actions = x
        self._n_transferts = int((x.shape[0]-1)/2)
        
    @property
    def n_transferts(self):
        return self._n_transferts
    
    def __repr__(self):
        return f"ActionSpace object, max transferts = {self.n_transferts}, values = {self.actions}"
    
    def __str__(self):
        return f"ActionSpace object, max transferts = {self.n_transferts}, values = {self.actions}"
    
#--- unitary tests for Action Space ------------------

# actions_space = ActionsSpace(n_transferts=3)
# print(actions_space)
# x = np.array( [-4,-3,-2,-1,0,1,2,3,4])
# actions_space.actions = x
# print(actions_space)

#--------------------------------------------------------------------------------
