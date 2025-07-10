# blackjack_env.py
import random

class BlackjackEnv:
    def __init__(
        self,
        use_counting=False,
        dealer_hits_soft_17=True,
        blackjack_payout=(1, 1),
        use_true_count=False,
        decks=1,
        use_bet_scaling=False,
        eps_floor=0.05,
        reward_shaping=False,
        use_basic_heuristic=False
    ):
        self.use_counting = use_counting
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.blackjack_payout = blackjack_payout
        self.use_true_count = use_true_count
        self.decks = decks
        self.use_bet_scaling = use_bet_scaling
        self.reward_shaping = reward_shaping
        self.use_basic_heuristic = use_basic_heuristic
        self.eps_floor = eps_floor
        self.running_count = 0
        self.action_space = [0, 1, 2]  # 0: Stand, 1: Hit, 2: Double Down
        self.can_double = True
        self.reset()

    def reset(self):
        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card(), self.draw_card()]
        self.done = False
        self.can_double = True  # Can double on first turn
        self.update_count(self.player + self.dealer)
        return self.get_obs(), {}

    def draw_card(self):
        card = random.randint(1, 10)
        self.update_count([card])
        return card

    def update_count(self, cards):
        if self.use_counting:
            for card in cards:
                if 2 <= card <= 6:
                    self.running_count += 1
                elif card == 10 or card == 1:
                    self.running_count -= 1

    def get_true_count(self):
        return self.running_count / self.decks if self.use_true_count else self.running_count

    def sum_hand(self, hand):
        total = sum(hand)
        if 1 in hand and total + 10 <= 21:
            return total + 10
        return total

    def is_bust(self, hand):
        return self.sum_hand(hand) > 21

    def is_blackjack(self, hand):
        return sorted(hand) == [1, 10]

    def get_obs(self):
        player_sum = self.sum_hand(self.player)
        dealer_up = self.dealer[0]
        
        # Check for usable ace (soft hand)
        has_usable_ace = 1 in self.player and player_sum != sum(self.player)
        
        # Basic state representation
        obs = (player_sum, dealer_up, has_usable_ace)
        
        if self.use_counting:
            true_count = self.get_true_count()
            # Discretize true count to reduce state space
            count_bucket = max(-5, min(5, int(round(true_count))))
            obs += (count_bucket,)
        
        return obs

    def get_bet(self):
        if self.use_bet_scaling:
            count = self.get_true_count()
            scaled_bet = 1 + max(0, int(count))
            return min(scaled_bet, 5)  # Cap max bet to 5 units
        return 1

    def step(self, action):
        if self.done:
            raise ValueError("Game has ended. Call reset().")

        bet = self.get_bet()

        if action == 1:  # Hit
            self.player.append(self.draw_card())
            self.can_double = False  # Can't double after hitting
            if self.is_bust(self.player):
                self.done = True
                reward = -1 * bet
                if self.reward_shaping and self.sum_hand(self.player[:-1]) <= 11:
                    reward += 0.2  # reward drawing when safe
                return self.get_obs(), reward, True, {"bet": bet, "action": "hit_bust"}
            else:
                return self.get_obs(), 0, False, {"bet": bet, "action": "hit"}
                
        elif action == 2 and self.can_double:  # Double Down
            self.player.append(self.draw_card())
            self.done = True
            bet *= 2  # Double the bet
            
            if self.is_bust(self.player):
                return self.get_obs(), -1 * bet, True, {"bet": bet, "action": "double_bust"}
            
            # Dealer plays
            while self.sum_hand(self.dealer) < 17 or (self.sum_hand(self.dealer) == 17 and self.dealer_hits_soft_17 and 1 in self.dealer):
                self.dealer.append(self.draw_card())
            
            player_score = self.sum_hand(self.player)
            dealer_score = self.sum_hand(self.dealer)
            
            if self.is_bust(self.dealer) or player_score > dealer_score:
                reward = 1 * bet
            elif player_score == dealer_score:
                reward = 0
            else:
                reward = -1 * bet
                
            return self.get_obs(), reward, True, {"bet": bet, "action": "double"}
            
        else:  # Stand (action == 0 or invalid double)
            if action == 2 and not self.can_double:
                # Invalid double, treat as stand
                pass
                
            while self.sum_hand(self.dealer) < 17 or (self.sum_hand(self.dealer) == 17 and self.dealer_hits_soft_17 and 1 in self.dealer):
                self.dealer.append(self.draw_card())
            self.done = True

            player_score = self.sum_hand(self.player)
            dealer_score = self.sum_hand(self.dealer)

            if self.is_bust(self.dealer) or player_score > dealer_score:
                if self.is_blackjack(self.player):
                    reward = self.blackjack_payout[0] / self.blackjack_payout[1]
                else:
                    reward = 1
            elif player_score == dealer_score:
                reward = 0
            else:
                reward = -1

            return self.get_obs(), reward * bet, True, {"bet": bet, "action": "stand"}