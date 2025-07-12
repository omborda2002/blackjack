import random

class BlackjackEnv:
    def __init__(self, use_counting=False, dealer_hits_soft_17=True, blackjack_payout=(3, 2),
                 use_true_count=False, decks=1, use_bet_scaling=False, eps_floor=0.05,
                 reward_shaping=False, toughest=False):
        self.use_counting = use_counting
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.blackjack_payout = blackjack_payout
        self.use_true_count = use_true_count
        self.decks = decks
        self.use_bet_scaling = use_bet_scaling
        self.reward_shaping = reward_shaping
        self.eps_floor = eps_floor
        self.running_count = 0
        self.cards_dealt = 0
        self.total_cards = 52 * decks
        self.action_space = [0, 1, 2]  # 0: Stand, 1: Hit, 2: Double
        self.bet = 1
        self.shuffle_deck()
        if toughest:
            self.apply_toughest_rules()
        self.reset()

    def apply_toughest_rules(self):
        self.blackjack_payout = (1, 1)
        self.dealer_hits_soft_17 = True
        self.decks = 8
        self.total_cards = 52 * self.decks
        self.use_bet_scaling = True
        self.shuffle_deck()

        # Additional tough rules
        self.restricted_double = True  # Only allow doubling on 9, 10, 11
        self.surrender_allowed = False
        self.dealer_wins_push = True
        self.penetration_limit = 0.75
        self.no_resplit_aces = True

    def shuffle_deck(self):
        self.deck = [rank for rank in range(1, 14)] * 4 * self.decks
        random.shuffle(self.deck)
        self.running_count = 0
        self.cards_dealt = 0

    def draw_card(self):
        if self.cards_dealt >= self.total_cards * 0.75:
            self.shuffle_deck()
        card = self.deck.pop()
        self.cards_dealt += 1
        if self.use_counting:
            self.running_count += self.card_count_value(card)
        return min(card, 10)

    def card_count_value(self, card):
        if 2 <= card <= 6:
            return 1
        elif card >= 10 or card == 1:
            return -1
        else:
            return 0

    def hand_value(self, hand):
        total, aces = 0, hand.count(1)
        for card in hand:
            total += 11 if card == 1 else card
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def get_true_count(self):
        remaining_decks = max(1, (self.total_cards - self.cards_dealt) / 52)
        return self.running_count / remaining_decks

    def reset(self):
        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card(), self.draw_card()]
        self.done = False
        self.can_double = True
        self.bet = 1

        if self.use_bet_scaling:
            true_count = self.get_true_count() if self.use_true_count else self.running_count
            self.bet = min(max(1, int(true_count)), 5)  # Cap bet between 1 and 5

        return self.get_obs()

    def get_obs(self):
        obs = (self.hand_value(self.player), self.dealer[0], 1 in self.player)
        if self.use_bet_scaling:
            count_bucket = min(5, max(-5, int(self.get_true_count())))
            return obs + (self.bet, count_bucket)
        if self.use_bet_scaling:
            return obs + (self.bet,)
        return obs

    def step(self, action):
        if self.done:
            return self.get_obs(), 0, True, {}

        if action == 1:  # Hit
            self.player.append(self.draw_card())
            if self.hand_value(self.player) > 21:
                self.done = True
                return self.get_obs(), -1 * self.bet, True, {"result": "bust"}
            return self.get_obs(), 0, False, {}

        elif action == 2 and self.can_double and (not hasattr(self, 'restricted_double') or self.hand_value(self.player) in [9, 10, 11]):  # Double
            self.player.append(self.draw_card())
            self.can_double = False
            self.done = True
            return self.resolve_hand(double=True)

        elif action == 0 or not self.can_double:  # Stand
            self.done = True
            return self.resolve_hand()
        else:
            return self.get_obs(), 0, False, {}

    def resolve_hand(self, double=False):
        while self.hand_value(self.dealer) < 17 or (self.dealer_hits_soft_17 and self.hand_value(self.dealer) == 17 and 1 in self.dealer):
            self.dealer.append(self.draw_card())

        player_score = self.hand_value(self.player)
        dealer_score = self.hand_value(self.dealer)
        reward = 0

        if player_score > 21:
            reward = -1
        elif dealer_score > 21 or player_score > dealer_score:
            reward = 1.5 if self.blackjack_payout == (3, 2) and len(self.player) == 2 and player_score == 21 else 1
        elif player_score == dealer_score:
            if hasattr(self, 'dealer_wins_push') and self.dealer_wins_push:
                reward = -1  # Push goes to dealer
            else:
                reward = 0
            reward = 0
        else:
            reward = -1

        if double:
            reward *= 2

        if self.reward_shaping:
            if reward > 0:
                reward += 0.1
            elif 'bust' in info.get('result', ''):
                reward -= 0.1
        return self.get_obs(), reward * self.bet, True, {"dealer_score": dealer_score, "player_score": player_score}