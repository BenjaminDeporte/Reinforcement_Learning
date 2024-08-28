#-----------------------------------------------------------------------------------------------
#--- RL for BlackJack --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

import numpy as np

#-----------------------------------------------------------------------------------------------
#--- basic classes -----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

#--- Card Iterator ----------------------------------------------------------------------------

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


#--- Deck ----

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
        self._usable_ace = None
        
    def __repr__(self):
        msg = "Deck :\n" + f"Player's cards = {self.player_cards}\n" + f"Dealer's cards = {self.dealer_cards}\n"
        return msg
    
    def __str__(self):
        msg = "Deck :\n" + f"Player's cards = {self.player_cards}\n" + f"Dealer's cards = {self.dealer_cards}\n"
        return msg
    
    def _draw_dealer(self):
        # draw a card for the dealer
        self.dealer_cards.append(next(self.card_iterator))
        
    def _draw_player(self):
        # draw a card for the player
        self.player_cards.append(next(self.card_iterator))
        
    def _calculate_player_sum(self):
        self._player_sum = 0
        hand_has_ace = False
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
        # à coder
        pass
                
            
#--- test

# for j in range(10):
#     deck = Deck()
#     for i in range(2):
#         print(deck)
#         # deck._draw_dealer()
#         deck._calculate_player_sum()
#         print(f"Player's sum = {deck._player_sum}, usable ace = {deck._usable_ace}")
#         deck._draw_player()