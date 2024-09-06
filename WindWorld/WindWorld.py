#----------------------------------------------------------------
#--- Windy GridWorld --------------------------------------------
#--- p131 Sutton and Barto --------------------------------------
#----------------------------------------------------------------

import numpy as np
import timeit

#----------------------------------------------------------------
#--- GLOBAL variables -------------------------------------------
#----------------------------------------------------------------

#--- size of the world
NX = 10
NY = 7

#--- wind conditions --------------------------------------------
WIND = np.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
)

#--- special states ---------------------------------------------
START = np.array([0,3])
GOAL = np.array([8,3])

#--- parameters -------------------------------------------------
ALPHA = 0.5
EPSILON = 0.1
GAMMA = 1.0

#--- randomness -------------------------------------------------
rng = np.random.default_rng(seed=42)

#----------------------------------------------------------------
#--- base class for Environment ---------------------------------
#----------------------------------------------------------------

class WindWorld():
    """class for Environment as described in example 6.5 p130
    """
    
    # four possible actions : up, down, right, left
    number_actions_classic = 4
    
    classical_actions = {
        0 : [0,1], # up
        1 : [0,-1], # down
        2 : [1,0], # right
        3 : [-1,0], # left
    }
    
    def __init__(self, nx=None, ny=None, wind=None, number_actions=None):
        """constructor

        Args:
            nx (_type_, optional): size in X. Defaults to NX.
            ny (_type_, optional): size in Y. Defaults to NY.
            wind (_type_, optional): array of Y-wind. Defaults to wind (global).
        """
        if nx == None:
            self.nx = NX
        else:
            self.nx = nx
        
        if ny == None:
            self.ny = NY
        else:
            self.ny = ny
            
        if wind == None:
            self.wind = WIND
        else:
            self.wind = wind
            
        if number_actions == None:
            self.number_actions = self.number_actions_classic
        else:
            self.number_actions = number_actions
            
        #--- q table ---------------------------------------------------------------
        self.q_table = np.zeros(shape=(self.nx, self.ny, self.number_actions))
        #--- init q(terminal,*) = 0 ---
        self.q_table[START[0],START[1],:] = np.array( [ 0 for a in range(self.number_actions)] )
        
        #--- epsilon greedy policy ----
        #--- [x,y,n_a] is probability of choosing action n_a when in state x,y
        self.greedy_policy = np.zeros(shape=(self.nx, self.ny, self.number_actions))
        
        #--- dynamics : probas to move to another state, and possible states
        #--- probabilities : maximum three possible arrival states (6.10 p131)
        #--- dynamics_probas[x,y,n_a,:] is the set of 3 probabilities for action n_a taken in state x,y
        self.dynamics_probas = np.zeros(shape=(self.nx, self.ny, self.number_actions, 3))
        # for classical windworld : deterministic policy
        for x in range(self.nx):
            for y in range(self.ny):
                self.dynamics_probas[x,y] = np.array([1,0,0])
        #--- arrival states
        #--- arrival_states[x,y,n_a,:, nx_s_prime, ny_s_prime] is the set of the three possible arrival states
        #--- when taking action n_a from state x,y
        self.arrival_states = np.zeros(shape=(self.nx, self.ny, self.number_actions, 3, 2))
        # for classical windworld : just one arrival state to compute
        for x in range(self.nx):
            for y in range(self.ny):
                for action_number in range(self.number_actions):
                    # get action moves
                    delta_x, delta_y = self.classical_actions.get(action_number)[0], self.classical_actions.get(action_number)[1]
                    arrival_x = x + delta_x
                    arrival_y = y + delta_y
                    # add the wind
                    arrival_y += self.wind[x]
                    # stay in bounds
                    if arrival_x < 0: arrival_x = 0
                    if arrival_y < 0: arrival_y = 0
                    if arrival_x >= self.nx: arrival_x = self.nx - 1
                    if arrival_y >= self.ny: arrival_y = self.ny - 1
                    # update arrival position : only one, with probability one
                    self.arrival_states[x,y,action_number,0] = np.array([arrival_x, arrival_y])
                    self.arrival_states[x,y,action_number,1] = None
                    self.arrival_states[x,y,action_number,2] = None
                       
        
    #--- news -----------------------------
    
    def __repr__(self):
        msg = f"Classic WindWorld object"
        msg += f"\nsize {self.nx} x {self.ny}"
        msg += f"\nwind = {np.array(self.wind)}"
        return msg
    
    #--- methods --------------------------
    
    #--- calculate epsilon greedy policy according to q_table ------------
    def _set_epsilon_greedy_policy(self, eps=0.1):
        """calculate greedy policy according to the current q_table, and update self.greedy_policy accordingly

        Args:
             eps (float, optional): epsilon parameter for the calculation. Defaults to 0.1.
        """
        
        proba_for_non_greedy_actions = eps / self.number_actions 
        proba_for_greedy_actions = 1 - eps + eps/self.number_actions
         
        for x in range(self.nx):
            for y in range(self.ny):
                # determine which action is greedy in x,y
                greedy_action_number = np.argmax(self.q_table[x,y,:])
                # update probabilities accordingly
                for action_number in range(self.number_actions):
                    if action_number == greedy_action_number:
                        self.greedy_policy[x,y,action_number] = proba_for_greedy_actions
                    else:
                        self.greedy_policy[x,y,action_number] = proba_for_non_greedy_actions
                        
    #--- choose action from state, according to policy -------------------
    def get_action_from_state(self, state):
        """choose an action from state according to policy

        Args:
            state (np.array([int,int])) : state coordinates
        """
        
        # get array of probabilities of the different possible actions from state
        action_probas = self.greedy_policy[state[0], state[1]]
        # form cdf
        cdf = 0
        # random
        random_action = rng.uniform()
        # find action number
        for a in range(self.number_actions):
            cdf += action_probas[a]
            if random_action <= cdf:
                return a
        # should not reach here
        raise NameError ("did not find an action")
    
    #--- is the state the terminal state ? -----------------
    def _is_terminal(self,x,y):
        return np.array_equal(np.array([x,y]), START)
    
    #--- get the SARSA action-----------------------------
    def get_sarsa(self, state, action_number):
        """get reward and arrival state from a state perfoming an action number

        Args:
            state (np.array): initial state x,y
            action_number (int): action number between 0 and self.number_actions
            
        Returns:
            arrival_state[0], arrival_state[1] : arrival state
            reward (int)
        """
        
        x = state[0]
        y = state[1]
        
        # choose random action number from state, 
        action_number = self.get_action_from_state(state)
        # get set of probas for next state (ie determinsitic in classical wind world)
        probas_arrival_states = self.dynamics_probas[x,y,action_number]
        cdf = 0
        # random
        random_uniform = rng.uniform()
        # find index id of arrival state
        for id in range(len(probas_arrival_states)):
            cdf += probas_arrival_states[id]
            if random_uniform <= cdf:
                break
        arrival_state = self.arrival_states[x,y,action_number,id]
        
        if self._is_terminal(arrival_state[0], arrival_state[1]):
            reward = 0
        else:
            reward = -1
            
        return arrival_state[0], arrival_state[1], reward
         


#--------------------------------------------------------------------

#--- tests --- 

# ww = WindWorld()

# print(ww)

# ww._set_epsilon_greedy_policy()

# # check q_table
# # print(ww.q_table)

# # check dynamics
# print(ww.dynamics_probas)

# # check arrival states
# print(f"arrival states for action UP")
# for x in range(ww.nx):
#     for y in range(ww.ny):
#         print(f"{x,y} UP => {ww.arrival_states[x,y,0,0]}")
        
# print(f"arrival states for action DOWN")
# for x in range(ww.nx):
#     for y in range(ww.ny):
#         print(f"{x,y} DOWN => {ww.arrival_states[x,y,1,0]}")
        
# print(f"arrival states for action RIGHT")
# for x in range(ww.nx):
#     for y in range(ww.ny):
#         print(f"{x,y} RIGHT => {ww.arrival_states[x,y,2,0]}")

# print(f"arrival states for action LEFT")
# for x in range(ww.nx):
#     for y in range(ww.ny):
#         print(f"{x,y} LEFT => {ww.arrival_states[x,y,3,0]}")

# print(ww.greedy_policy)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- SARSA -----------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# ww = WindWorld()

# N_EPISODES = 10
# N_STEPS = 10

# # --- loop on episodes -------------------
# for episode in range(N_EPISODES):
#     state = START
#     # update greedy policy
#     ww._set_epsilon_greedy_policy()
#     # get action number from state, according to policy
#     action_number = ww.get_action_from_state(state)
    
#     #-- loop steps in the episode
#     for step in range(N_STEPS):
#         # take action action_number from state, observe reward and arrival state
#         arrival_state_x, arrival_state_y, r = ww.get_sarsa(state, action_number)
#         # choose action a' from arrival_state s', using policy from Q
#         action_prime_number = ww.get_action_from_state(np.array([arrival_state_x, arrival_state_y]))
#         # update q_table
#         ww.q_table[state[0], state[1], action_number] += ALPHA * ( r + GAMMA * ww.q_table[arrival_state_x, arrival_state_y,action_prime_number])
#         # get ready for next loop
#         state[0] = arrival_state_x
#         state[1] = arrival_state_y
#         action_number = action_prime_number
#         if ww._is_terminal(state[0], state[1]) is True:
#             # end episode if terminal state is reached
#             break
        