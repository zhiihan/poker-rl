import numpy as np
import logging


class Card:
    def __init__(self, num: int, suit: int):
        self.suit = suit
        self.num = num

        assert 2 <= num <= 14, "Not a valid number"
        assert 0 <= suit <= 3, "Not a valid suit"

        self.disp = {i: str(i) for i in range(2, 11)}
        self.disp.update({i: j for i, j in zip(range(11, 15), ["J", "Q", "K", "A"])})
        self.disp_suits = {
            i: str(j)
            for i, j in zip(range(0, 4), ["\u2660", "\u2665", "\u2666", "\u2663"])
        }

    def __lt__(self, other):
        return self.num < other.num

    def __repr__(self):
        return self.disp[self.num] + self.disp_suits[self.suit]

    def __str__(self):
        return self.disp[self.num] + self.disp_suits[self.suit]


class PokerGame:
    def __init__(self, num_players: int = 1):
        self.num_cards = 52
        self.num_players = num_players
        self.cards = [Card(num=i, suit=s) for i in range(2, 15) for s in range(0, 4)]

        np.random.shuffle(self.cards)

        self.community_cards = self.cards[-5:].copy()
        self.cards = self.cards[5:].copy()

        self.stage = [False, False, False]  # Flop, Turn, River

        self.player_cards = []
        for i in range(num_players):
            self.player_cards.append(self.cards[i * 2 : i * 2 + 2])

    def shuffle_deck(self):
        np.random.shuffle(self.cards)

    def start(self):
        pass

    def end(self):
        pass

    def evaluate_hand(self, cards_list: list[Card]):
        """
        Evaluate the player's hand.

        This function should take the player's hand,
        combine it with the community cards, and output the Poker rank.

        Args:
            cards_list: A list of Cards possibly exceeding 5.

        Returns:
            rank: The rank of the cards.
            best_hand: The best hand possible for the player.
        """

        disp_rank = {
            1: "Five of a Kind",
            2: "Straight Flush",
            3: "Four of a Kind",
            4: "Full House",
            5: "Flush",
            6: "Straight",
            7: "Three of a Kind",
            8: "Two Pair",
            9: "One Pair",
            10: "High Card",
        }

        cards_list = sorted(cards_list)

        rank, best_hand = self.check_kinds(cards_list)

        straight = self.check_straight(cards_list)
        if straight and rank > 6:
            rank = 6
            best_hand = straight

            flush_straight = self.check_flush(straight)
            if flush_straight and rank > 2:
                rank = 2
                best_hand = flush_straight

        flush = self.check_flush(cards_list)
        if flush and rank > 5:
            rank = 5
            best_hand = flush

        print(f"Your best hand is {disp_rank[rank]}, {best_hand}")

    def check_straight(self, cards_list):
        """
        Check for straight.

        Returns:
            The best straight possible.
        """

        consecutive = []

        for card in cards_list[::-1]:
            if not consecutive:
                consecutive.append(card)
            elif card.num == consecutive[-1].num:
                continue
            elif card.num + 1 == consecutive[-1].num:
                consecutive.append(card)
            else:
                consecutive = [card]

            if len(consecutive) >= 5:
                best_hand = consecutive[:5]
                # print("Straight", best_hand)
                return best_hand

    def check_flush(self, cards_list):
        """
        Check for flush.

        Returns:
            The best flush possible.
        """
        # Check for flush
        suits = np.array([0, 0, 0, 0])

        for card in cards_list:
            suits[card.suit] += 1
        if any(suits >= 5):
            flush_suit = np.argmax(suits)

            best_hand = sorted([card for card in cards_list if card.suit == flush_suit])

            # print("Flush", best_hand[-5:])
            return best_hand[-5:]

    def check_kinds(self, cards_list):
        """
        Check for X of a kind, Full house/Two Pair.
        """

        # Check for matches
        multiples = []
        matches = []
        max_len = 1
        for card in cards_list:
            if not matches:
                matches.append(card)
            elif matches[0].num == card.num:
                matches.append(card)
                max_len = max(len(matches), max_len)
            elif matches[0].num != card.num:
                multiples.append(matches)
                matches = [card]
        multiples.append(matches)
        multiples = sorted(multiples, key=len)

        if max_len <= 5:
            # Convert kind to the Poker Rank. 5 of a kind == highest, ...
            kind_mapping = {5: 1, 4: 3, 3: 7, 2: 9, 1: 10}
            rank = kind_mapping[max_len]

        else:
            # Unphysical, more than 5 of a kind
            rank = 0

        largest_match = multiples[-1]

        best_hand = largest_match
        second_max_len = 1
        for pairs in multiples[-2::-1]:
            if len(pairs) + len(best_hand) < 5:
                second_max_len = max(len(pairs), second_max_len)
                best_hand.extend(pairs)
            elif len(pairs) + len(best_hand) == 5:
                second_max_len = max(len(pairs), second_max_len)
                best_hand.extend(pairs)
                break

        if second_max_len == 2 and max_len == 3:
            rank = 4
            # print("Full House", best_hand)
        elif second_max_len == 2 and max_len == 2:
            rank = 8
            # print("Two Pair", best_hand)
        else:
            # print(f"{max_len} of a kind", best_hand)
            pass

        return rank, best_hand
