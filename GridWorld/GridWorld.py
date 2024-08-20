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
# actions
actions = {
    0 : (0,-1), # up
    1 : (0,+1), # down
    2 : (+1,0), # right
    3 : (-1,0), # left
}
NUM_ACTIONS = len(actions)

# --- basic classes --------------------------------------------------------------------------

# - Value Function -------

class ValueFunction():
    """Value function class. Stores value functions for each state, provides basic get, update and display methods
    """
    
    def __init__(self, nx=NX, ny=NY, value_function=None):
        assert 0<nx and 0<ny and isinstance(nx,int) and isinstance(ny, int), f"Erreur paramètres constructeur ValueFunction"
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

# - Policy -----------------

class Policy():
    """Policy class. Stores probabilities of each action (up, down, right, left) per state.
    """
    action_to_str = {
        0 : "U",
        1 : "D",
        2 : "R",
        3 : "L"
    }
    
    def __init__(self, nx=NX, ny=NY, policy=None):
        assert 0<nx and 0<ny and isinstance(nx,int) and isinstance(ny, int), f"Erreur paramètres constructeur Policy"
        self.nx = nx
        self.ny = ny
        if policy is None:
            self.policy = np.full(shape=(self.nx, self.ny, NUM_ACTIONS), fill_value=1/NUM_ACTIONS)  # default is equiprobable random policy
        else:
            self.policy = policy
            
    def get(self, x,y):
        assert (0 <= x < self.nx) and (0 <= y < self.ny), "erreur : hors grid dans Policy.get()"
        return self.policy[x,y]
    
    def update(self, x,y, value):
        # value is a np.array shape NUM_ACTIONS x 1
        assert (0 <= x < self.nx) and (0 <= y < self.ny), "erreur : hors grid dans Policy.update()"
        self.policy[x,y] = value
        
    def display(self):
        print(self.policy)
        
    def get_graphic_display(self):
        chars = np.full(shape=(self.nx, self.ny), dtype=object, fill_value="")
        for x in range(self.nx):
            for y in range(self.ny):
                local = self.get(x,y) # get local policy
                msg = ""
                for action_number in actions.keys():
                    value = local[action_number]
                    msg = msg + self.action_to_str.get(action_number) + f"({value:.2f})"
                chars[x,y] = msg
        return np.transpose(chars)  # transpose because of the x,y coordinates convention
                

    def __repr__(self):
        msg = f"Objet Policy taille {self.nx} x {self.ny} x {NUM_ACTIONS} - shape = {self.policy.shape}"
        return msg
        
    def __str__(self):
        msg = f"Objet Policy taille {self.nx} x {self.ny} x {NUM_ACTIONS} - shape = {self.policy.shape}"
        return msg
        
# -- MDP Dynamics ----------------

class MDPDynamics():
    """Code the dynamics of the MDP. 
    For GridWorld, this is a deterministic dynamic : the next state is reached with probability one
    """
    
    def __init__(self, actions=actions):
        self.actions = actions
        
    def step(self, x,y, action_number):
        """calculate next step

        Args:
            x, y (ints) : coordinates of the current state
            action_number (int): code of the action
            
        Returns :
            x_new, y_new (ints): coordinates of the state being reached
            reward (int): reward associated to the move
            end (boolean) : True if terminal state is reached
        """
        step_x, step_y = actions.get(action_number)
        
        x_new = x + step_x
        if x_new < 0: x_new = 0
        if x_new >= NX: x_new = NX-1
        
        y_new = y + step_y
        if y_new < 0: y_new = 0
        if y_new >= NY: y_new = NY-1
        
        if (x_new, y_new)==(0,0) or (x_new,y_new)==(NX-1,NY-1):
            end = True
            reward = -1
        else:
            end = False
            reward = -1
        
        return x_new, y_new, reward, end
    
    def __repr__(self):
        msg = f"Objet MDPDynamics. Actions = {self.actions}"
        return msg
    
    def __str__(self):
        msg = f"Objet MDPDynamics. Actions = {self.actions}"
        return msg
    
# ------------------------------------------------------------------------------------------------------
# --- Policy Evaluation --------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

class IterativePolicyEvaluation():
    """Calculate one iteration step of a value function towards the optimal value function v*
    """
    # THETA = 1e-6 # threshold to stop iterating
    dynamics = MDPDynamics()
    
    def __init__(self, policy, vf_old=None):
        """Constructor, to evaluate a given <policy> iteratively starting from a ValueFunction <v_start>

        Args:
            policy (Policy): object Policy to evaluate.
            v_start (ValueFunction, optional): ValueFunction to use as a start of the algorithm. Defaults to None, in which case 0 is used
        """
        # policy to evaluate
        self.policy = policy
        # value function to use as a first iteration
        if vf_old is None:
            self.vf_old = ValueFunction()  # the default ValueFunction is 0 for all states
        else:
            self.vf_old = vf_old
        # store first iteration for record
        self.vf_start = self.vf_old
        # value function calculation for policy, place holder
        self.vf_new = ValueFunction()
            
    def evaluation_step(self):
        """Return one step evaluation of the policy"""
        for x in range(NX):
            for y in range(NY):
                # state s is (x,y)
                if (x,y) != (0,0) and (x,y) != (NX-1,NY-1):  # update value function for non terminal states only
                    for action_number in actions.keys():
                        # action is action_number
                        # get s' and r
                        x_new, y_new, reward, end = self.dynamics.step(x,y,action_number)
                        # update vf_new(x,y)
                        self.vf_new.vf[x,y] += self.policy.get(x,y)[action_number] * (reward + GAMMA * self.vf_old.vf[x_new, y_new])
        # calculate Norm 2 between the update and the original
        delta_vf = np.linalg.norm(self.vf_new.vf - self.vf_old.vf)
        
        return self.vf_new, delta_vf

# ----------------------------------------------------------------------------------------------------------    
# --- Calcul de la value function optimale v* sans optimisation de la policy -------------------------------
# ----------------------------------------------------------------------------------------------------------
        
# iter_counter = 0
# THETA = 1e-12
# delta_vf = 2 * THETA
# vf_old = ValueFunction()   # instantiate a ValueFunction equal to zero for all states
# random_policy = Policy()
# ipe = IterativePolicyEvaluation(random_policy)

# print(f"Value Function avant itération :")
# vf_old.display()

# while delta_vf > THETA:
#     iter_counter += 1
#     vf_evaluation, delta_vf = ipe.evaluation_step()
#     if iter_counter % 100 == 0:
#         print(f"Iteration {iter_counter}")
#         print(f"Value Function après calcul :")
#         vf_evaluation.display()
#         print(f"Norme 2 = {delta_vf:.7f}")
#     ipe.vf_old = vf_evaluation
#     ipe.vf_new = ValueFunction()

# print("\n")
# print(f"Calcul de la value function optimale à la précision {THETA} après {iter_counter} itérations")
# vf_evaluation.display()


# --------------------------------------------------------------------------------------------------------------
# --- Policy Improvement ---------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

class PolicyImprovement():
    """Given a policy and a value function, return an improved policy, or signals the policy is already optimal
    """
    
    dynamics = MDPDynamics()
    
    def __init__(self, policy, value_function):
        """instantiate the object with a given policy and a given value_function

        Args:
            policy (Policy): the starting policy, to improve
            value_function (ValueFunction): the value function to use to improve the policy
        """
        self.start_policy = policy
        self.new_policy = Policy()
        
        self.start_vf = value_function
        
    def improvement_step(self):
        """logic to improve the policy. Returns optimal=True if policy already optimal
        """
        
        optimal = True
        
        for x in range(NX):
            for y in range(NY):
                # state s is (x,y)
                if (x,y) != (0,0) and (x,y) != (NX-1,NY-1):  # update policy for non terminal states only
                    # first, find out all four q_values for each of the four actions
                    current_potential_q_values = np.zeros(shape=NUM_ACTIONS)
                    for action_number in actions:
                        # nouvel état suite à action
                        x_new, y_new, reward, end = self.dynamics.step(x,y,action_number)
                        # value function au nouvel état
                        v_value = self.start_vf.get(x_new, y_new)
                        # calcul q_value correspondante
                        current_potential_q_values[action_number] = reward + GAMMA * v_value
                    v_max = np.max(current_potential_q_values)
                    idx = np.array([ 1 if current_potential_q_values[i]==v_max else 0 for i in actions.keys() ])  # 1 for action getting max value
                    new_pol = idx / np.sum(idx) # normalize to get probabilities
                    self.new_policy.update(x,y,new_pol)  # write new policy
                    if self.new_policy != self.start_policy:
                        optimal = False   # current policy is not optimal if there is a change
                        
        return optimal, self.new_policy
        
# ------------------------------------------------------------------------------------------------------------------
# --- Policy Improvement (testing one step) ------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# policy = Policy() # random policy to start

# ipe = IterativePolicyEvaluation(policy)
# value_function, _ = ipe.evaluation_step()   # calculate value function for random policy

# print(f"Start :")
# print(f"Policy = ")
# print(policy.get_graphic_display())
# print(f"Value function :")
# value_function.display()

# policy_improvement = PolicyImprovement(policy, value_function)  # starting point : random policy with associated value function
# optimal, new_policy = policy_improvement.improvement_step()

# print(f"Stop :")
# print(f"Policy = ")
# print(new_policy.get_graphic_display())

# -----------------------------------------------------------------------------------------------------------------
# --- Policy Iteration --------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# CODE HERE
















# -- unitary tests ------------------------------------------------------------------------------------------------

# --- MDP Dynamics -----------------------------------------

# dyn = MDPDynamics()

# print(dyn)

# positions = [
#     [0,0], [0,1], [1,1], [3,3], [2,3]
# ]

# for position in positions:
#     for action_code in actions.keys():
#         x_new, y_new, reward, end = dyn.step(position[0], position[1], action_code)
#         print(f"Step : depuis {position[0], position[1]} avec action {action_code} - Résultat : nouvelle position {x_new, y_new}, fini = {end}, reward = {reward}")

# ---- Policy ------

# pi = Policy()

# print(pi)

# pi.display()