#------------------------------------------------------------------
#--- Jack's Car Rental --------------------------------------------
#------------------------------------------------------------------

#--- toy case Sutton & Barto ch4 p81 ------------------------------

#--- librairies ---------------------------------------------------

import numpy as np
import math
import timeit

#--- parameters ---------------------------------------------------

MAX_CARS = 5 # maximum number of cars at each rental location
MAX_TRANSFERTS = 3  # maximum number of cars that can be moved overnight
GAMMA = 0.9 # discount
LAMBDA_CUSTOMERS_1 = 3  # Poisson law parameter for customer requests at location 1
LAMBDA_CUSTOMERS_2 = 4  # Poisson law parameter for customer requests at location 2
LAMBDA_RETURNS_1 = 3 # Poisson law parameter for cars returns at location 1
LAMBDA_RETURNS_2 = 2 # Poisson law parameter for cars returns at location 1
UNITARY_TRANSFERT_COST = 2  # cost of moving one car overnight
UNITARY_RENTAL_PRICE = 10  # revenue for renting one car

#--- base utilities -----------------------------------------------

#--- Poisson laws : calculate log probas from 0 to N_CARS included -------
#--- NB : probas for n > N_CARS are assumed negligible -------------------

customers_1 = np.array([ n*np.log(LAMBDA_CUSTOMERS_1) - np.log(math.factorial(n))-LAMBDA_CUSTOMERS_1 for n in range(MAX_CARS+1)])
customers_2 = np.array([ n*np.log(LAMBDA_CUSTOMERS_2) - np.log(math.factorial(n))-LAMBDA_CUSTOMERS_2 for n in range(MAX_CARS+1)])
returns_1 = np.array([ n*np.log(LAMBDA_RETURNS_1) - np.log(math.factorial(n))-LAMBDA_RETURNS_1 for n in range(MAX_CARS+1)])
returns_2 = np.array([ n*np.log(LAMBDA_RETURNS_2) - np.log(math.factorial(n))-LAMBDA_RETURNS_2 for n in range(MAX_CARS+1)])

#-------------------------------------------------------------------------------
#--- base classes --------------------------------------------------------------
#-------------------------------------------------------------------------------

#--- here, the choice is not to use Gym, for learning purposes -----------------

#-------------------------------------------------------------------------------
#--- class for state space -----------------------------------------------------
#-------------------------------------------------------------------------------

# class StatesSpace():
#     """Encapsulates states space as a N_CARS x N_CARS array, plus some display methods
#     Constructor requires n_cars maximum per location
#     """
    
#     def __init__(self, max_cars=None):
#         """Create a StateSpace object, basically a np.array(n_cars x n_cars)

#         Args:
#             max_cars (_type_, optional): maximum number of cars per agency. Defaults to None.
#         """
#         if max_cars is None:
#             self._max_cars = MAX_CARS
#         else:
#             self._max_cars = max_cars
            
#         # by convention : 
#         # -the numbers of cars at location #1 (end of business day) is the first coordinate
#         # -the numbers of cars at location #2 (end of business day) is the second coordinate
#         # ie : [n1, n2]
#         self._states = np.zeros((self.max_cars, self.max_cars), dtype=int)
        
#     @property
#     def max_cars(self):
#         return self._max_cars
    
#     @max_cars.setter
#     def max_cars(self, n):
#         self._max_cars = n
    
#     @property
#     def states(self):
#         return self._states
    
#     @states.setter
#     def states(self, array_of_states):
#         self._states = array_of_states
        
#     def display(self):
#         print (self.states)
        
#     def __repr__(self):
#         return f"StateSpace object, size {self.max_cars} x {self.max_cars}"
    
#     def __str__(self):
#         return f"StateSpace object, size {self.max_cars} x {self.max_cars}"
    
#--- unitary test for StateSpace -----

# states_space = StatesSpace(max_cars = 5)
# print(states_space.states)
# states_space.n_cars = 8
# x = np.ones((states_space.n_cars, states_space.n_cars))
# states_space.states = x
# states_space.display()

#--------------------------------------------------------------------------------
#--- class for action space -----------------------------------------------------
#--------------------------------------------------------------------------------

# class ActionsSpace():
#     """Ecapsulates data and methods regarding actions.
#     Constructor requires n_transferts, which is the maximum number of cars that can be moved overnight.
#     By defintion, the number of cars moved from location 1 to location 2 is counted positive.
#     """
    
#     def __init__(self, max_transferts=None):
#         """Create an action space object, basically a np.array(2*n_transferts+1)

#         Args:
#             max_transferts (_type_, optional): maximum number of cars that can be moved overnight. Defaults to None.
#         """
#         if max_transferts == None:
#             self._max_transferts = MAX_TRANSFERTS  # default value
#         else:
#             self._max_transferts = int(max_transferts)
            
#         # actions vary from -max 
#         self._actions = np.array( [ n for n in range(-self._max_transferts, self._max_transferts+1)] )
        
#     @property
#     def actions(self):
#         return self._actions
    
#     @actions.setter
#     def actions(self, x):
#         self._actions = x
#         self._max_transferts = int((x.shape[0]-1)/2)
        
#     @property
#     def max_transferts(self):
#         return self._max_transferts
    
#     def __repr__(self):
#         return f"ActionSpace object, max transferts = {self.max_transferts}, values = {self.actions}"
    
#     def __str__(self):
#         return f"ActionSpace object, max transferts = {self.max_transferts}, values = {self.actions}"
    
#--- unitary tests for Action Space ------------------

# actions_space = ActionsSpace(max_transferts=3)
# print(actions_space)
# x = np.array( [-4,-3,-2,-1,0,1,2,3,4])
# actions_space.actions = x
# print(actions_space)

#--------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#--- transition logic given state, action and daily business and returns ----------------
#----------------------------------------------------------------------------------------
    
def transition(state, action, daily_numbers):
    """Basic transition logic. 
    Takes a state (n1, n2) at end of business day, an action (number of cars to transfer overnight), and a set of daily events (cars requests and returns during the day).
    Calculate the end state (n1,n2) at enf of next day, and the reward

    Args:
        state (np.array(2,1)): state at end of business day, ie [n1,n2] with n1 number of cars at location 1, n2 number of cars at location 2.
        action (int): action. number of cars to transfer from location 1 to location 2. Must be between -MAX_TRANSFERTS and +MAX_TRANSFERTS
        daily_numbers (np.array(4,1))): array (B1,B2,R1,R2) with : B1 number of cars requests at location 1, B2 number of cars requests at location 2, R1 number of returns at location 1, R2 number of returns at location 2.
        
    Returns:
        new_state (np.array(2,1)) : state after tranferts and processing of daily business
        reward : business return of the day
    """
    
    # init reward
    reward = 0
    # get numbers of cars end of previous day
    n1_t = state[0]
    n2_t = state[1]
    # execute overnight transfer
    if action > n1_t:
        raise NameError(f"Location 1 has {n1_t} cars but {action} are requested to transfer")
    if action < -n2_t:
        raise NameError(f"Location 2 has {n2_t} cars but {action} are requested to transfer")
    n1 = n1_t - action
    n2 = n2_t + action
    reward -= np.abs(action) * UNITARY_TRANSFERT_COST
    # calculate number of cars being rented during day
    B1 = daily_numbers[0]
    rent1 = min(n1, B1) # can not rent more than the stock
    B2 = daily_numbers[1]
    rent2 = min(n2, B2) # can not rent more than the stock
    reward += (rent1 + rent2) * UNITARY_RENTAL_PRICE
    # get returns during the day
    R1 = daily_numbers[2]
    R2 = daily_numbers[3]
    # calculate stocks end of current day, capped at MAX_CARS
    # cast as int because used as indexes for array
    n1_t_plus_1 = int(min( n1 - rent1 + R1, MAX_CARS))
    n2_t_plus_1 = int(min( n2 - rent2 + R2, MAX_CARS))
    # format outputs
    new_state = np.array([n1_t_plus_1, n2_t_plus_1])
    
    return new_state, reward

#------------------------------------------------------------------------------------------------------
#--- unitary tests ------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
        
#--- transition function -----

#--- random cases

# rng = np.random.default_rng()
# SAMPLES = 10

# states_list = [
#     np.array([ rng.choice(MAX_CARS+1), rng.choice(MAX_CARS+1) ]) for i in range(SAMPLES) 
# ]

# possible_actions = np.arange(-MAX_TRANSFERTS, MAX_TRANSFERTS+1)
# actions_list = [
#     rng.choice(possible_actions) for i in range(SAMPLES)
# ]

# plage = np.arange(0,6)
# daily_numbers_list = [
#     np.array(
#         [ rng.choice(plage), rng.choice(plage), rng.choice(plage), rng.choice(plage)]
#     ) for i in range(SAMPLES)
# ]

# for state, action, daily_numbers in zip(states_list, actions_list, daily_numbers_list):
#     print()
#     print(f"state = {state}")
#     print(f"action = {action}")
#     print(f"daily numbers : requests_1 = {daily_numbers[0]}, requests_2 = {daily_numbers[1]}, returns_1 = {daily_numbers[2]}, returns_2 = {daily_numbers[3]}")
#     new_state, reward = transition(state, action, daily_numbers)
#     print(f"new state = {new_state}")
#     print(f"reward = {reward}")

#--------------------------------------------------------------------------------
#--- classe pour Policy ---------------------------------------------------------
#--------------------------------------------------------------------------------

class DeterministicPolicy():
    """This is the class to manage a deterministic policy:
    - holds records of every action (ie number of cars to transfer overnight) per given state
    - sanitization to ensure tranfers are possible (ie do not go beyond numbers of cars)
    - holds the iterative algorithm to converge to the policy value function
    """
    
    THETA = 1e-6   # convergence criterion
    IMPROVEMENT_THRESHOLD = 1e-9  # check improvement in two successive value functions
    
    # --- constructor ------------------------------------------------------------------------
    def __init__(self, actions_array=None):
        
        # the policy is an array of MAX_CARS x MAX_CARS of number of cars to transfer overnight from location 1 to location 2
        if actions_array is None:
            # if no policy is given, init to 0 (no cars transfered)
            self._actions = np.zeros((MAX_CARS+1, MAX_CARS+1))
        else:
            # if a policy is given,
            # check shape
            assert actions_array.shape == (MAX_CARS+1, MAX_CARS+1), "Wrong policy shape passed to DeterministicPolicy constructor"
            # store the policy
            self._actions = actions_array
            # sanitize policy
            sanitized = self._sanitize_actions()
            if sanitized:
                print(f"action array got clipped in DeterministicPolicy constructor")

        # place holder for policy value funtion (to calculate)
        self._policy_value_function = None
        
        # value functions arrayS for the iterative calculation
        self._old_vf = np.zeros((MAX_CARS+1, MAX_CARS+1))
        self._new_vf = np.zeros((MAX_CARS+1, MAX_CARS+1))
        
    # --- info ----
    
    def __repr__(self):
        return f"Object DeterministicPolicy, action array shape = {self.actions.shape}, policy evaluated : {self._policy_value_function is not None}"
    
    def __str__(self):
        return f"Object DeterministicPolicy, action array shape = {self.actions.shape}, policy evaluated : {self._policy_value_function is not None}"

    # --- sanitizor -------------------------------------------------------------------------
    def _sanitize_actions(self):
        # check every action is possible given locations' cars stocks
        sanitized = False
        for n1 in range(MAX_CARS+1):
            for n2 in range(MAX_CARS+1):
                # get projected number of cars to transfer from location 1 with n1 to location 2 with n2
                current_action = self._actions[n1,n2]
                # check transfert is possible, if not, clip number
                if current_action > n1 :
                    self._actions[n1,n2] = n1
                    sanitized = True
                if current_action < -n2:
                    self._actions[n1,n2] = -n2
                    sanitized = True
        return sanitized
    
    # --- get, set functions --------------------------------------------------------------------------
    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, x):
        assert x.shape == (MAX_CARS+1, MAX_CARS+1), "Wrong policy shape passed to actions setter in DeterministicPolicy object"
        self._actions = x
        sanitized = self._sanitize_actions()
        if sanitized:
            print(f"action array got clipped when used to set a DeterministicPolicy")
    
    @property
    def policy_value_function(self):
        # if self._policy_value_function is None:
        self._policy_evaluation()
            
        return self._policy_value_function
    
    @policy_value_function.setter
    def policy_value_function(self):
        raise NameError(f"Attempt to write a policy evaluation directly in a DeterministicPolicy object")
    
    # @policy_value_function.setter
    # def policy_value_function(self):
    #     raise NameError(f"Attempt to write directly a value function in DeterministicPolicy object")
    
    # --- one policy evaluation step ------------------------------------------------------------------
    
    def _evaluation_step(self):
        """perform one step of policy evaluation, update self._new_vf
        """
        
        # init
        old_vf = self._old_vf
        new_vf = np.zeros_like(old_vf)
        
        # check feasibility of policy for every state
        sanitized = self._sanitize_actions()
        if sanitized is True:
            print(f"Policy was sanitized (ie some tranferts were clipped) prior to policy evaluation")
        
        # number of sweeps
        number_sweeps_to_perform = (MAX_CARS+1)**6
        number_sweeps_performed = 0
        # perform ONE sweep
        for n1 in range(MAX_CARS+1):
            for n2 in range(MAX_CARS+1):
                # get starting state
                state = np.array([n1,n2])
                # get the policy action planned for state (n1,n2), feasibility has been checked above
                policy_action = self.actions[n1,n2]
                # envision all possible business events -----------------------------------------------
                # we consider only business rentals requests up to MAX_CARS
                for B1 in range(MAX_CARS+1):
                    # get log proba of having B1 requets according to the Poisson law
                    log_pB1 = customers_1[B1] 
                    for B2 in range(MAX_CARS+1):
                        log_pB2 = customers_2[B2]  
                        # consider only returns up to MAX_CARS
                        for R1 in range(MAX_CARS+1):  
                            log_pR1 = returns_1[R1] 
                            for R2 in range(MAX_CARS+1):
                                log_pR2 = returns_2[R2]
                                # calculate total probability of all four events, assumed independent of course
                                log_p = log_pB1 + log_pB2 + log_pR1 + log_pR2
                                # calculate end state
                                daily_numbers = np.array([B1,B2,R1,R2])
                                new_state, reward = transition(state, policy_action, daily_numbers)
                                # calculate delta for value function
                                delta_vf = np.exp(log_p) * ( reward + GAMMA * old_vf[new_state[0], new_state[1]])
                                # update value function
                                new_vf[n1,n2] = new_vf[n1,n2] + delta_vf
                                # update
                                number_sweeps_performed += 1
                                # print(f"calculated {number_sweeps_performed} expected returns / {number_sweeps_to_perform}", end="\r")
                                
        # at this point, one sweep has been performed and the next iteration of value function wrt old_vf has been computed in new_vf
        self._new_vf = new_vf
        
    # --- full policy evaluation ----------------------------------------------------------------------
    
    def _policy_evaluation(self):
        """Evaluate policy. Iterations until convergence
        """
        
        convergence_criterion = 2 * self.THETA
        
        # inits
        # start from 0 policy (no transferts)
        self._old_vf = np.zeros((MAX_CARS+1, MAX_CARS+1))
        # counting
        iteration_number = 1
        # print(f"starting policy evaluation")
        # loop
        while convergence_criterion > self.THETA:
            # print(f"iteration number : {iteration_number} -----------------------------------")
            # perform one step
            self._evaluation_step()
            convergence_criterion = np.max(np.abs(self._old_vf - self._new_vf))
            self._old_vf = self._new_vf
            iteration_number += 1
            print(f"Iteration {iteration_number} - Norm inf convergence criterion = {convergence_criterion:.2e}", end="\r")
        # iteration is complete
        print()
        self._new_vf = self._old_vf
        self._policy_value_function = self._new_vf
        
    # --- policy improvement --------------------------------------------------------------------------
    
    def _policy_improvement(self):
        """Perform a one-step policy improvement of a current policy with an associated value function
        """
        
        # get the value function of the policy (NB : assumed to be calculated already)
        pvf = self._policy_value_function
        
        # inits
        # get the current policy (ie action per state) and place holder for improved policy
        # check feasibility of policy for every state
        sanitized = self._sanitize_actions()
        if sanitized is True:
            print(f"Policy was sanitized (ie some tranferts were clipped) prior to policy improvement")
        old_policy = self.actions
        new_policy = np.zeros((MAX_CARS+1, MAX_CARS+1))
        
        # change flag
        optimized = False
        
        # loop
        # number of sweeps
        number_sweeps_to_perform = (MAX_CARS+1)**6 * (2*MAX_TRANSFERTS+1)
        number_sweeps_performed = 0
        # perform ONE sweep
        for n1 in range(MAX_CARS+1):
            for n2 in range(MAX_CARS+1):
                # get starting state
                state = np.array([n1,n2])
                # get current value function of the state and current action
                current_vf = pvf[n1,n2]
                current_action = old_policy[n1,n2]
                # try all actions and calculate their q values
                q_values = np.zeros(2*MAX_TRANSFERTS+1)
                for action in range(-MAX_TRANSFERTS, +MAX_TRANSFERTS+1):
                    # skip impossible actions
                    if action > n1: 
                        number_sweeps_performed += (MAX_CARS+1)**4
                        continue
                    if action < -n2: 
                        number_sweeps_performed += (MAX_CARS+1)**4
                        continue
                    # calculate q_value of action considered
                    q_value = 0
                    # we consider only business rentals requests up to MAX_CARS
                    for B1 in range(MAX_CARS+1):
                        # get log proba of having B1 requets according to the Poisson law
                        log_pB1 = customers_1[B1] 
                        for B2 in range(MAX_CARS+1):
                            log_pB2 = customers_2[B2]  
                            # consider only returns up to MAX_CARS
                            for R1 in range(MAX_CARS+1):  
                                log_pR1 = returns_1[R1] 
                                for R2 in range(MAX_CARS+1):
                                    log_pR2 = returns_2[R2]
                                    # calculate total probability of all four events, assumed independent of course
                                    log_p = log_pB1 + log_pB2 + log_pR1 + log_pR2
                                    # calculate end state
                                    daily_numbers = np.array([B1,B2,R1,R2])
                                    new_state, reward = transition(state, action, daily_numbers)
                                    # get q value for starting state and envisoned action
                                    q_value += np.exp(log_p) * ( reward + GAMMA * pvf[new_state[0], new_state[1]] )
                                    # update
                                    number_sweeps_performed += 1
                                    print(f"calculated {number_sweeps_performed} situations / {number_sweeps_to_perform}", end="\r")
                    q_values[action+MAX_TRANSFERTS] = q_value
                # find argmax q_values and check if better
                max_q_value = np.max(q_values)
                if max_q_value > current_vf + self.IMPROVEMENT_THRESHOLD:  
                    # yes, there is a q_value better than the current value_function : improve policy !
                    id_argmax = np.argmax(q_values)
                    action_max = id_argmax - MAX_TRANSFERTS
                    new_policy[n1,n2] = action_max
                    # signal that policy has been strictly improved
                    optimized = True
        print()
        
        # calculate infinite norm between the old and new value functions
        gain = np.max(np.abs(new_policy - old_policy))
        
        # policy is considered improved if it has changed AND value function has increased above a threshold                                    
        if (optimized is True and gain > self.IMPROVEMENT_THRESHOLD):
            # print(f"\nPolicy has improved")
            self.actions = new_policy
        else:
            optimized = False
            # print(f"\nPolicy is optimal")

        return optimized
        
    
    
#------------------------------------------------------------------------------------------------------
#--- unitary tests : class DeterministicPolicy --------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# --- constructor ---

# dp = DeterministicPolicy()

# print(dp)
# print(dp.actions)
# print(dp._policy_value_function)
# print(dp._old_vf)
# print(dp._new_vf)

# --- sanitize ----

# rng = np.random.default_rng()
# random_actions = np.random.randint(low=-3, high=MAX_CARS+5, size=(MAX_CARS, MAX_CARS))

# print(random_actions)
# dp = DeterministicPolicy(actions_array=random_actions)
# print(dp.actions)

# --- one evaluation step ------------------------------

# dp = DeterministicPolicy()  # instantiate with 0 transferts Policy

# dp._evaluation_step()

# --- full evaluation -----------------------------------

# dp = DeterministicPolicy()

# print(dp.actions)

# print (f"premier accès")
# start = timeit.default_timer()
# print(dp.policy_value_function)
# duration = timeit.default_timer() - start
# print(f"Policy evaluation done in {duration:.2f} seconds")

# print(f"deuxième accès")
# print(dp.policy_value_function)

# --- policy improvement ------------------------------------------------

dp = DeterministicPolicy()

optimized = False
iter = 1

print(f"--- Policy Iteration for Jack's Car Rental --------")
print(f"Maximum number of cars at each location : {MAX_CARS}")
print(f"Maximum number of transferts overnight : {MAX_TRANSFERTS}")
print()

with np.printoptions(precision=3, suppress=True):
    while True:
        print(f"\nIteration {iter}")
        print("Current policy is:")
        print(dp.actions)
        print(f"Evaluating current policy (ie calculate policy's value function)...")
        print(dp.policy_value_function)
        print(f"... value function calculated")
        print(f"Try to improve policy...")
        optimized = dp._policy_improvement()
        if optimized is False:
            print(f"...Current policy is optimal")
            break
        print(f"... calculated a better policy")
        iter += 1

    print(f"\nEnd of iterations")
    print(f"Optimal policy found")
    print(dp.actions)
    print(f"Value function:")
    print(dp.policy_value_function)