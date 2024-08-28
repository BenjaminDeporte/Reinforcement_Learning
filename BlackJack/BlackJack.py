#-----------------------------------------------------------------------------------------------
#--- RL for BlackJack --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

import numpy as np

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
        self.rng = np.random.default_rng()
        
    def __iter__(self):
        # return the iterator object
        return self
    
    def __next__(self):
        # return a rancom card
        couleur = self.colors.get(self.rng.choice(len(self.colors)))
        figure = self.figures.get(self.rng.choice(len(self.figures)))
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

class MCES(self):
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
    ACTIONVALUE_SHAPE = (10,2,10,2)
    
    # policy array
    STATEVALUE_SHAPE = (10,2,10)
    
    # rewards
    WIN = 1
    LOSS = -1
    DRAW = 0
    
    # parameters
    NUMBER_RUNS_PER_STATE_ACTION_PAIR = 10
    NUMBER_RANDOM_STATE_ACTION_PAIRS = 5
    
    def __init__(self):
        self._action_value_table = np.zeros(shape=self.ACTIONVALUE_SHAPE)
        self._state_value_table = np.zeros(shape=self.STATEVALUE_SHAPE)
        self.rng = np.random.Generator(seed=42)
        
    def one_run(self, deck, action):
        """perform one run (episode) out of a (deck,action) pair and calculate one return.
        deck : class Deck
        action : "HIT" or "STICK"
        """

        player_sum, usable_ace, dealer_sum = deck.state
        # record starting state and action
        initial_player_sum, initial_usable_ace, initial_dealer_sum = deck.state
        # record initial action
        initial_action = action
        
        # generate episode
        episode_return = 0
        
        # phase 0 : check if player has a BlackJack ! ----------------------------------------------
        if deck.player_sum == 21:
            # player has a natural, ie Blackjack
            # draw a second card for the dealer
            deck._draw_dealer()
            if deck.dealer_sum == 21:
                episode_return = self.DRAW
            else:
                episode_return = self.WIN
                
        # # phase 1 : player hits until player's sum is greater or equal to 12 ---------------
        # below = (player_sum <= 11)
        # while below:
        #     deck._draw_player()
        #     below = (deck.player_sum <= 11)
        