#-----------------------------------------------------------------------------------------------
#--- RL for BlackJack --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
import pickle
import os

#-----------------------------------------------------------------------------------------------
#--- basic classes -----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#--- Card Iterator ----------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class CardIterator():
    """class Iterable and Iterator. Returns random cards.
    """
    
    colors = {
        0 : "Carreau",
        1 : "Coeur",
        2 : "Pique",
        3 : "Trèfle"
    }
    
    figures = {
        0 : "As",
        1 : 2,
        2 : 3,
        3 : 4,
        4 : 5,
        5 : 6,
        6 : 7,
        7 : 8,
        8 : 9,
        9 : 10,
        10 : "Valet",
        11 : "Dame",
        12 : "Roi" 
    }
    
    def __init__(self):
        pass
        # self.rng = global_rng
        
    def __iter__(self):
        # return the iterator object
        return self
    
    def __next__(self):
        # return a rancom card
        couleur = self.colors.get(global_rng.choice(len(self.colors)))
        figure = self.figures.get(global_rng.choice(len(self.figures)))
        return (couleur, figure)
    
#--- test

# ci = CardIterator()

# ctr = 0
# for carte in ci:
#     print(carte)
#     ctr += 1
#     if ctr > 10:
#         break

#----------------------------------------------------------------------------------------------
#--- Deck -------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

class Deck():
    """encapsulates player's cards and dealer's cards, calculate sums
    """
    
    figures_value = {
        "As" : 0,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 5,
        6 : 6,
        7 : 7,
        8 : 8,
        9 : 9,
        10 : 10,
        "Valet" : 10,
        "Dame" : 10,
        "Roi" : 10
    }
    
    def __init__(self):
        self.card_iterator = CardIterator()
        # draws 2 cards for player
        self.player_cards = [ next(self.card_iterator) for i in range(2)]
        # draw one visible card for dealer
        self.dealer_cards = [ next(self.card_iterator) ]
        # dealer show
        _, self._dealer_visible_card = self.dealer_cards[0]
        # init sums
        self._player_sum = None
        self._dealer_sum = None
        # flag useable as
        self._usable_ace = False
        
    def __repr__(self):
        msg = "Deck :\n" + f"Player's cards = {self.player_cards}\n" + f"Dealer's cards = {self.dealer_cards}"
        return msg
    
    def __str__(self):
        msg = "Deck :\n" + f"Player's cards = {self.player_cards}\n" + f"Dealer's cards = {self.dealer_cards}"
        return msg
    
    def _draw_dealer(self):
        # draw a card for the dealer
        self.dealer_cards.append(next(self.card_iterator))
        
    def _draw_player(self):
        # draw a card for the player
        self.player_cards.append(next(self.card_iterator))
        
    def _calculate_player_sum(self):
        # calculate the sum of cards of the player
        self._player_sum = 0
        hand_has_ace = False
        self._usable_ace = False
        for _, figure in self.player_cards:
            if figure == "As":
                # count As as one for starter, will check later if usable or not
                hand_has_ace = True
                self._player_sum += 1
            else:
                self._player_sum += self.figures_value.get(figure)
        # check if usable ace
        if hand_has_ace == True:
            if self._player_sum <= 11:
                self._player_sum += 10
                self._usable_ace = True
            else:
                self._usable_ace = False
                
    def _calculate_dealer_sum(self):
        # calculate the sum of cards of the dealer
        self._dealer_sum = 0
        hand_has_ace = False
        for _, figure in self.dealer_cards:
            if figure == "As":
                # count As as one for starter, will check later if usable or not
                hand_has_ace = True
                self._dealer_sum += 1
            else:
                self._dealer_sum += self.figures_value.get(figure)
        # check if usable ace
        if hand_has_ace == True:
            if self._dealer_sum <= 11:
                self._dealer_sum += 10
                # self._usable_ace = True
            else:
                # self._usable_ace = False
                pass
            
    def _get_dealer_visible_card(self):
        # return value of dealer visible card : 1 for Ace, 2 to 9, 10 for 10 and figure
        _, card = self.dealer_cards[0]
        if card == "As":
            val = 1
        else:
            if card in [10, "Valet", "Dame", "Roi"]:
                val = 10
            else:
                val = int(card)
        self._dealer_visible_card = val
                
    # getters
    @property
    def player_sum(self):
        self._calculate_player_sum()
        return self._player_sum
    
    @property
    def dealer_sum(self):
        self._calculate_dealer_sum()
        return self._dealer_sum
    
    @property
    def state(self):
        return self.player_sum, self._usable_ace, self.dealer_sum
    
    @property
    def dealer_visible_card(self):
        self._get_dealer_visible_card()
        return self._dealer_visible_card
                
            
#--- test

# for j in range(10):
#     deck = Deck()
#     for i in range(2):
#         print(deck)
#         print(f"Player's sum = {deck.player_sum}, usable ace = {deck._usable_ace}")
#         print(f"State = {deck.state}")
#         deck._draw_dealer()

#----------------------------------------------------------------------------------------------------------
#--- Game Simulator ---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

# ----- for tests only ----------------------------------------
def basic(player_sum, usable_ace, dealer_sum):
    """One example of policy

    Args:
        player_sum (_type_): player's points
        usable_ace (_type_): As being used at 11 (True/False)
        dealer_sum (_type_): visible dealer's points

    Returns:
        string : HIT or STICK
    """
    if player_sum <= 16:
        return "HIT"
    else:
        return "STICK"
    

    
class Game():
    """class that simulates an entire game, returns the reward at the end of the game
    """
    
    # rewards
    WIN = 1
    LOSS = -1
    DRAW = 0
    
    def __init__(self, verbose=False, policy=basic):
        # create an initial deck
        self.verbose = verbose
        self.deck = Deck()
        if self.verbose:
            print(f"Initial deck :")
            print(self.deck)
        self.policy = basic
            
    def set_deck(self):
        """set deck with a given list of cards
        """
        # à coder
        pass
            
    def run(self):
        """run the entire game
        """
        
        # phase 0 : check if player has a BlackJack ! ----------------------------------------------
        if self.deck.player_sum == 21:
            # player has a natural, ie Blackjack
            # draw a second card for the dealer
            self.deck._draw_dealer()
            if self.verbose:
                print(f"Final deck:")
                print(deck)
            if self.deck.dealer_sum == 21:
                if self.verbose: print(f"Player has blackjack, dealer too! DRAW")
                return self.DRAW, self.deck.player_sum, self.deck.dealer_sum
            else:
                if self.verbose: print(f"Player has blackjack, dealer doesn't, WIN")
                return self.WIN, self.deck.player_sum, self.deck.dealer_sum
            
        # phase 1 : draws additional player cards if initial two cards sum below 11 ---------------
        below = (self.deck.player_sum <= 11)
        auto_hits = False
        while below:
            auto_hits = True
            self.deck._draw_player()
            if self.verbose:
                print(f"Player sum below 11, drew another card")
            below = (self.deck.player_sum <= 11)
        if self.verbose:
            if auto_hits == False:
                print(f"No auto hit")
            else:
                print(f"Auto hits were performed")
                print(f"Deck after auto-hits:")
                print(self.deck)
                
        # phase 2 : player's turn
        # get state : player's sum, useable ace (True/False), dealer's sum
        player_sum, usable_ace, dealer_sum = self.deck.state
        # get policy (HIT or STICK) based on state and policy
        action = self.policy(player_sum, usable_ace, dealer_sum)
        while action == "HIT":
            if self.verbose:
                print(f"Player action is HIT")
            # get another card
            self.deck._draw_player()
            player_sum, usable_ace, dealer_sum = self.deck.state
            if self.verbose:
                print(f"Deck after HIT : {self.deck}")
            if player_sum > 21:
                # player goes bust
                if self.verbose: print(f"Player goes bust with {player_sum}: LOSS")
                return self.LOSS, self.deck.player_sum, self.deck.dealer_sum
            action = self.policy(player_sum, usable_ace, dealer_sum)
        # player is done playing
        if self.verbose:
            print(f"Player is done drawing cards")
            print(self.deck)
            
        # phase 3 : dealer's turn
        if self.verbose:
            print(f"Dealer's turn")
        player_sum, usable_ace, dealer_sum = self.deck.state
        while dealer_sum <= 16:
            self.deck._draw_dealer()
            if self.verbose:
                print(f"Dealer draws a card")
                print(self.deck)
            player_sum, usable_ace, dealer_sum = self.deck.state
        if self.verbose:
            print(f"Dealer is done drawing cards")
        if dealer_sum > 21:
            if self.verbose:
                print(f"Dealer has gone bust with {dealer_sum} : WIN")
                return self.WIN, self.deck.player_sum, self.deck.dealer_sum
            
        # phase 4 : conclusion
        if player_sum == dealer_sum: reward = self.DRAW
        if player_sum > dealer_sum: reward = self.WIN
        if player_sum < dealer_sum: reward = self.LOSS
        return reward, player_sum, dealer_sum
    
    
# --- test --------

# game = Game(verbose=True)

# r,p,d = game.run()

# print(f"Result is {r} - player has {p}, dealer has {d}")


#---------------------------------------------------------------------------------------------------
#--- MONTE CARLO with EXPLORING STARTS -------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

class MCES():
    """class to encapsulate the Monte Carlo Exploring Starts algo
    """
    
    # action value array
    # --- axis 0 to 2 : state -----------------
    # first axis is number of possible player's sums : from 12 to 21 included, total is 10 possibilities
    # second axis is whether a player has a usable ace or not : 0 for no, 1 for yes. Total is 2 possibilities
    # third axis is number of possible dealer's visible card : As, 2 to 9, figure : total is 10 possibilities
    # --- axis 3 : action ---------------------
    # fourth axis is action : 0 is HIT, 1 is STICK
    # --- value ----
    # value is estimate of Q(s,a)
    # shape is player_sum x usable_ace x dealer_sum x action
    ACTIONVALUE_SHAPE = (10,2,10,2)
    
    # policy array
    # shape is player_sum x usable_ace x dealer_sum
    STATEVALUE_SHAPE = (10,2,10)
    
    # rewards
    WIN = 1
    LOSS = -1
    DRAW = 0
    
    # per default parameters
    NUMBER_RUNS_PER_STATE_ACTION_PAIR = 100
    NUMBER_RANDOM_STATE_ACTION_PAIRS = 1000
    
    def __init__(self, number_random_state_action_pairs=None, number_runs_per_state_action_pair=None, verbose=None):
        
        # set total number of state/action pairs to sweep
        if number_random_state_action_pairs is None:
            self.number_random_state_action_pairs = self.NUMBER_RANDOM_STATE_ACTION_PAIRS
        else:
            self.number_random_state_action_pairs = number_random_state_action_pairs
            
        # set number of runs to perform for each state/action pair
        if number_runs_per_state_action_pair is None:
            self.number_runs_per_state_action_pair = self.NUMBER_RUNS_PER_STATE_ACTION_PAIR
        else:
            self.number_runs_per_state_action_pair = number_runs_per_state_action_pair
            
        if verbose is None:
            self.verbose = False
        else:
            self.verbose = True
            
        # initiate tables
        # self._action_value_table = np.zeros(shape=self.ACTIONVALUE_SHAPE)
        # self._state_value_table = np.zeros(shape=self.STATEVALUE_SHAPE)
        
        # set up RNG
        # self.rng = global_rng
        
        
    def run(self):
        """full MCES run
        """
        
        # inits
        
        # arbitrary policy -------------------
        # shape is player_sum x usable_ace x dealer_sum
        # first dim : player sum varies between 12 and 21, is coded at index 0 to 9
        # second dim : usable ace is index 0, non-usable ace is index 1
        # third dim : dealer visible card is As (index 0), 2 to 9 (index 1 to 8), or value 10 (index 9)
        # value : 0 is HIT, 1 is STICK
        policy = np.zeros(shape=(10,2,10))
        
        # action-state table ------------------
        # first three axis same as policy
        # fourth axis is player action : HIT at index 0, STICK at index 1
        q_table = np.zeros(shape=(10,2,10,2))
        
        print(f"-------------------------------------------------")
        print(f"---- learning -----------------------------------")
        print(f"-------------------------------------------------")
        print(f"-- sweeping {self.NUMBER_RANDOM_STATE_ACTION_PAIRS} action-state pairs ---")
        print(f"--- {self.number_runs_per_state_action_pair} times each ---- ")
        print(f"-------------------------------------------------")
        print()
        
        # first loop : sweep through random action-state pairs
        for number_action_pair_state in range(self.number_random_state_action_pairs):
            # randomly choose a state/action pair
            # draw cards, choose first action
            start_deck = Deck()
            # 0 is HIT, 1 is STICK
            first_action = global_rng.choice(2)
            list_returns = []
            # second loop : several runs per action-state pair
            for number_runs in range(self.number_runs_per_state_action_pair):
                # donne des nouvelles
                print(f"Action-Pair numéro {number_action_pair_state}, run numéro {number_runs} / {self.number_runs_per_state_action_pair}", end="\r")
                # deep copy to avoid shallow copies
                deck = deepcopy(start_deck)
                # first check : does the player has a Blackjack ?
                if deck.player_sum == 21:
                    # player has a natural, ie Blackjack
                    # draw a second card for the dealer
                    deck._draw_dealer()
                    # check if dealer has a 21 too
                    if deck.dealer_sum == 21:
                        episode_return = self.DRAW
                    else:
                        episode_return = self.WIN
                    list_returns.append(episode_return)
                    continue # next episode
                else:
                    # no blackjack, draw cards for player up to >= 12
                    below = (deck.player_sum <= 11)
                    auto_hits = False
                    while below:
                        auto_hits = True
                        deck._draw_player()
                        if self.verbose:
                            print(f"Player sum below 11, drew another card")
                        below = (deck.player_sum <= 11)
                    if self.verbose:
                        if auto_hits == False:
                            print(f"No auto hit")
                        else:
                            print(f"Auto hits were performed")
                            print(f"Deck after auto-hits:")
                            print(deck)
                            
                    # then, apply first action
                    if first_action == 0: # HIT
                        deck._draw_player()
                        player_sum, usable_ace, dealer_sum = deck.state
                        if player_sum > 21:
                            # player goes bust
                            episode_return = self.LOSS
                            list_returns.append(episode_return)
                            continue # next episode
                            
                    # then apply policy : player draws when policy action is HIT
                    player_sum, usable_ace, dealer_sum = deck.state
                    dealer_show = deck.dealer_visible_card
                    if usable_ace is True:
                        id_usable_ace = 0
                    else:
                        id_usable_ace = 1
                    policy_action = policy[player_sum-12, id_usable_ace, dealer_show-1]
                    get_out = False
                    while policy_action == 0:
                        deck._draw_player()
                        player_sum, usable_ace, dealer_sum = deck.state
                        dealer_show = deck.dealer_visible_card
                        if player_sum > 21:
                            # player goes bust
                            episode_return = self.LOSS
                            list_returns.append(episode_return)
                            get_out = True
                            break # get out of while loop
                        if usable_ace is True:
                            id_usable_ace = 0
                        else:
                            id_usable_ace = 1
                        policy_action = policy[player_sum-12, id_usable_ace, dealer_show-1]

                    if get_out is True:
                        # exited from while, then exit to next loop
                        continue
                    
                    # then, dealer's turn
                    if self.verbose:
                        print(f"Dealer's turn")
                    player_sum, usable_ace, dealer_sum = deck.state
                    while dealer_sum <= 16:
                        deck._draw_dealer()
                        if self.verbose:
                            print(f"Dealer draws a card")
                            print(deck)
                        player_sum, usable_ace, dealer_sum = deck.state
                    if self.verbose:
                        print(f"Dealer is done drawing cards")
                    if dealer_sum > 21:
                        episode_return = self.WIN
                        list_returns.append(episode_return)
                        continue # next episode
                    else:
                        # phase 4 : conclusion
                        player_sum, usable_ace, dealer_sum = deck.state
                        if player_sum == dealer_sum:
                            episode_return = self.DRAW
                        if player_sum > dealer_sum:
                            episode_return = self.WIN
                        if player_sum < dealer_sum:
                            episode_return = self.LOSS
                        list_returns.append(episode_return)
                        continue                        
                        
                        
            # here, list_returns is the list of returns of random episodes with the starting action-state pair
            # calculate average return
            avg_return = sum(list_returns) / len(list_returns)
            # update q-table
            player_sum, usable_ace, dealer_sum = start_deck.state
            dealer_show = deck.dealer_visible_card
            if usable_ace is True:
                id_usable_ace = 0
            else:
                id_usable_ace = 1
            q_table[player_sum-12, id_usable_ace, dealer_show-1, first_action] = avg_return
            # update policy
            greedy_policy = np.argmax(q_table[player_sum-12, id_usable_ace, dealer_show-1,:])
            policy[player_sum-12, id_usable_ace, dealer_show-1] = greedy_policy
            # reporting 
            if number_action_pair_state % 100 == 1:
                print("-------------------------------------------------------")
                print(f"Initial deck :")
                print(start_deck)
                act = "HIT" if first_action == 0 else "STICK"
                print(f"First action : {act}")
                # print(f"Returns = {list_returns}")
                print(f"Average return = {avg_return}")
                best_first = "HIT" if greedy_policy == 0 else "STICK"
                print(f"Best first action : {best_first}")
            
        return q_table, policy
            
# test

global_rng = np.random.default_rng()

mces = MCES()

q_table, policy = mces.run()

# visualisation

print()
print(f"-------------------------------------------------")
print(f"--------- some examples after learning  ---------")
print(f"-------------------------------------------------")

N_SAMPLES = 10

for s in range(N_SAMPLES):
    print(f"Sample {s}")
    deck = Deck()
    print(deck)
    player_sum, usable_ace, dealer_sum = deck.state
    dealer_show = deck.dealer_visible_card
    if usable_ace is True:
        id_usable_ace = 0
    else:
        id_usable_ace = 1
    best_policy = np.argmax(q_table[player_sum-12, id_usable_ace, dealer_show-1,:])
    best_first = "HIT" if best_policy == 0 else "STICK"
    print(f"Best first action : {best_first}")
    state_value = np.max(q_table[player_sum-12, id_usable_ace, dealer_show-1,:]) 
    print(f"State value : {state_value}")
    
    
# saves results

dir_path = os.getcwd()

q_file = dir_path + "/BlackJack/q_table_file"
with open(q_file,"wb") as f:
    pickle.dump(q_table, f)
    
# display

dir_path = os.getcwd()
q_file = dir_path + "/BlackJack/q_table_file"
with open(q_file, "rb") as f:
    q_table = pickle.load(f)
    
print(q_table.shape)

usable_ace_q_table = q_table[:,1,:,:]
# print(usable_ace_q_table.shape)
ua_policy = np.full(shape=(10,10), fill_value="")
for i in range(10):
    for j in range(10):
        p = usable_ace_q_table[i,j]
        if p[0] > p[1]:
            ua_policy[i,j] = "H"
        else:
            ua_policy[i,j] = "S"
            
print(f"---------------------------------------")
print(f"-- usable ace policy ------------------")
print(f"---------------------------------------")
print(ua_policy)

not_usable_ace_q_table = q_table[:,0,:,:]
# print(not_usable_ace_q_table.shape)
not_ua_policy = np.full(shape=(10,10), fill_value="")
for i in range(10):
    for j in range(10):
        p = not_usable_ace_q_table[i,j]
        if p[0] > p[1]:
            not_ua_policy[i,j] = "H"
        else:
            not_ua_policy[i,j] = "S"
            
print(f"---------------------------------------")
print(f"-- no usable ace policy ---------------")
print(f"---------------------------------------")
print(not_ua_policy)