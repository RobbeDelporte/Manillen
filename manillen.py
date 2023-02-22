import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import pygame
import os
import time

SUITS = ["SPADES","HEARTS","DIAMONDS","CLUBS"]
RANKS = [7,8,9,11,12,13,1,10]

def get_image(path):
    from os import path as os_path
    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc

def env(render_mode=None):

    env = raw_env(render_mode=render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "manillen"}

    def __init__(self, render_mode=None):

        self.screen = None
        self.render_mode = render_mode

        self.possible_agents = ["player_" + str(r) for r in range(4)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        self.action_spaces = {agent: Discrete(32) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=0,high=1,shape=(164,),dtype=np.uint8),
                "action_mask": Box(low=0, high=1,shape=(32,),dtype=np.uint8),
            }) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    def reset(self, seed=None, return_info=False, options=None):
        self.hands = np.zeros((4,32),dtype=np.uint8)
        self.board = np.zeros((4,32),dtype=np.uint8)
        self.winning_agent_number = None
        self.first_agent_number = None
        # keep tracks of points until all 8 rounds are played then calculate rewards
        self.points = np.zeros((4,1))
        self.troef = 0
        self.points = np.zeros((4,1))

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
   
        self.agent_selection = self.agents[0]

        # shuffle deck
        deck = np.arange(32)
        if seed:
            np.random.seed(seed)
        np.random.shuffle(deck)
        for i, card in enumerate(deck):
            self.hands[i//8][card] = 1
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def legal_moves(self,agent):
        # i work with agent number instead of name
        agent_number = self.agent_name_mapping[agent]
        
        # I assume the agent == active_agent, otherwise this code doesnt make sense
        hand = self.hands[agent_number]
        # because of the representation of the board, the sum of it will always represent the turn number
        turn = np.sum(self.board, axis=None)
        # if the player plays first all cards are legal
        if turn == 0: 
            return hand
        
        # from vector to integer
        first_card = self.board[self.first_agent_number].argmax()
        winning_card = self.board[self.winning_agent_number].argmax()

        # Whether or not the teammate is winning or not influences legal cards
        team_winning = False
        if (self.winning_agent_number == 0 or self.winning_agent_number == 2) and (agent_number == 0 or agent_number == 2):
            team_winning = True
        if (self.winning_agent_number == 1 or self.winning_agent_number == 3) and (agent_number == 1 or agent_number == 3):
            team_winning = True

        # cards in the hand of the agent that follow leading card or beat winning card
        following_cards = np.zeros(32,dtype=np.uint8)
        winning_cards = np.zeros(32,dtype=np.uint8)
        for card in range(32):
            if hand[card] == 1:
                # card follows if it is of the same suit
                if card//8 == first_card//8:
                    following_cards[card] = 1

                # card wins current set if it wins over the currently winning card
                # card wins over card if it is uniquely troef or has higher rank and same suit
                if card//8 == self.troef and winning_card//8 != self.troef:
                    winning_cards[card] = 1
                elif card//8 == winning_card//8 and card%8 > winning_card%8:
                    winning_cards[card] = 1

        # if team is winning al following cards are legal
        if team_winning:
            # if no following cards, all cards are legal
            if np.sum(following_cards,axis=None) == 0:
                return hand
            return following_cards

        # if team not winning
        # just trust me that this logic conforms to the game rules (i hope)
        winning_following_cards = np.logical_and(following_cards,winning_cards)*1
        if np.sum(winning_following_cards,axis=None) > 0:
            return winning_following_cards
        if np.sum(following_cards,axis=None) > 0:
            return following_cards
        if np.sum(winning_cards,axis=None) > 0:
            return winning_cards
        return hand   

    def observe(self, agent):
        agent_number = self.agent_name_mapping[agent]

        # observation of one agent is its current hand, (features of) the board, troef
        hand = self.hands[agent_number]
        board = self.board.flatten()
        troef = np.array([1 if i == self.troef else 0 for i in range(4)],dtype=np.uint8)
        observation = np.concatenate([hand,board,troef],axis=0,dtype=np.uint8)

        legal_moves = self.legal_moves(agent) if agent == self.agent_selection else []
        action_mask = np.zeros(32,dtype=np.uint8)
        for i in range(len(legal_moves)):
            if legal_moves[i]:
                action_mask[i] = 1
    
        return {"observation": observation, "action_mask": action_mask}

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_number = self.agent_name_mapping[agent]
        legal_moves = self.legal_moves(agent)

        assert legal_moves[action] == 1, "played illegal move."

        turn = np.sum(self.board, axis=None)

        # add card to board
        self.board[agent_number][action] = 1
        # remove card from player hand
        self.hands[agent_number][action] = 0

        # update first_agent and winning_agent
        if turn == 0:
            self.first_agent_number = agent_number
            self.winning_agent_number = agent_number
        else:
            winning_card = self.board[self.winning_agent_number].argmax()
            if action//8 == self.troef and winning_card//8 != self.troef:
                self.winning_agent_number = agent_number
            elif action//8 == winning_card//8 and action%8 > winning_card%8:
                self.winning_agent_number = agent_number

        # set next agent, will be overwritten if new set starts
        self.agent_selection = self.agents[(agent_number + 1) % 4]

        # if every agent played their turn, happens when turn was 3 and last agent submitted action
        if turn == 3:
            for card in np.argwhere(self.board==1)[:,1]:
                if card%8 >= 3:
                    self.points[self.winning_agent_number] += (card%8 - 2)
            self.agent_selection = self.agents[self.winning_agent_number]
            # clear board
            self.board = np.zeros((4,32),dtype=np.uint8)
    
        # check if last set is played, if all cards are played
        if np.sum(self.hands,axis=None) == 0:
            # give rewards to winning team
            p1 = self.points[0] + self.points[2]
            p2 = self.points[1] + self.points[3]      
            self.rewards[self.agents[0]] += int((p1>30)*(p1-30))
            self.rewards[self.agents[1]] += int((p2>30)*(p2-30))
            self.rewards[self.agents[2]] += int((p1>30)*(p1-30))
            self.rewards[self.agents[3]] += int((p2>30)*(p2-30))

            # terminate all agents, game is done
            self.terminations = {i: True for i in self.agents}

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        screen_width = 700
        screen_height = 400

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.event.get()
        elif self.screen is None:
            self.screen = pygame.Surface((screen_width, screen_height))

        font = pygame.font.Font(pygame.font.get_default_font(), 40)
        self.screen.fill((0,0,0))

        for agent in range(4):
            # cards in hand
            cards = np.argwhere(self.hands[agent]==1)[:,0]
            for cc, card in enumerate(cards):
                suit = card//8
                rank = card%8
                card_name = str(RANKS[rank]) + SUITS[suit] + ".png"
                card_img = get_image(os.path.join("img/cards", card_name))
                self.screen.blit(card_img,[cc*50+10,agent*100+10])

            # card on board, should be either 1 or 0
            card = np.argwhere(self.board[agent]==1)[:,0]
            if len(card) == 1:
                card = card[0]
                suit = card//8
                rank = card%8
                card_name = str(RANKS[rank]) + SUITS[suit] + ".png"
                card_img = get_image(os.path.join("img/cards", card_name))
                self.screen.blit(card_img,[500,agent*100+10])

            points_img = font.render(str(self.points[agent]),True,(255,255,255))
            self.screen.blit(points_img,[600,agent*100+25])

        if self.render_mode == "human":
            pygame.display.update()
            


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None