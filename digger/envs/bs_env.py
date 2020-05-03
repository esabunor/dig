import random
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

INITIAL_ACCOUNT_BALANCE = 20000.0
MAX_ACCOUNT_BALANCE = 2000000.0
MAX_STEPS = 100
MAX_CURRENCY_PRICE = 500
MAX_VOLUME = 1000


class BSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df=None, training=False):
        self.df = df
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.nav = INITIAL_ACCOUNT_BALANCE
        self.max_nav = INITIAL_ACCOUNT_BALANCE
        self.realizedPL = 0
        self.unrealizedPL = 0
        self.trades = 0
        self.buy_price = 0
        self.sell_price = 0
        self.position_size = 0
        self.training = training
        # observation space - ohlc of the past 12 ticks (an hour)
        # action space - buy at most 20% of asset net asset, with a take profit of 60% or wait
        # self.observation_space =
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=((MAX_STEPS + 1) * 5,))

        self.current_step = random.randint(MAX_STEPS, len(self.df.loc[:, 'Open'].values) - MAX_STEPS)
        self.initial_step = self.current_step
        self.previous_action = None  # buy

    def step(self, action):
        # Execute one time step within the environment
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "Close"]
        self.current_step += 1

        self._take_action(action, current_price)

        if self.current_step > len(self.df.loc[:, 'Close'].values) - MAX_STEPS:
            self.current_step = MAX_STEPS
        reward = 0

        trade = False
        if self.previous_action == None:
            trade = True
        else:
            if (action == 0 and self.previous_action == 1) or (action == 1 and self.previous_action == 0):
                trade = True

        if trade:
            if self.previous_action == 0:
                # long
                price_diff = current_price - self.buy_price
                reward += price_diff * 1000
            elif self.previous_action == 1:
                # short
                price_diff = self.sell_price - current_price
                reward += price_diff * 1000
            elif self.previous_action == None:
                self.previous_action = action

        # reward = (self.balance + self.unrealizedPL) * delay_modifier
        # reward = self.balance * delay_modifier
        done = self.nav <= 0
        obs = self._next_observation()
        info = {
            'current_step': self.current_step,
            'reward': reward,
            'balance': self.balance,
            'position_size': self.position_size,
            'nav': self.nav,
            'unrealizedPL': self.unrealizedPL,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'position': "Long" if action == 0 else "Short"
        }
        return obs, reward, done, info

    def _take_action(self, action, current_price):
        # take profit to stop loss ratio is 3:1
        trade = False

        if self.previous_action == None:
            trade = True
        else:
            if (action == 0 and self.previous_action == 1) or (action == 1 and self.previous_action == 0):
                trade = True

        if trade:
            if self.previous_action:
                # not the first action
                if action == 0:
                    # buy
                    # previous action is sell so calculate profit for the short position
                    self.unrealizedPL = (self.sell_price - current_price) * self.position_size
                    self.buy_price = current_price
                    self.position_size = 0.2 * self.balance * 100
                elif action == 1:
                    # sell
                    self.unrealizedPL = (current_price - self.buy_price) * self.position_size  #  calculate profit for previous postion
                    self.sell_price = current_price
                    self.position_size = 0.2 * self.balance * 100
                self.previous_action = action
                self.balance = self.nav + self.unrealizedPL
                self.realizedPL += self.unrealizedPL
                self.nav = self.balance
                self.unrealizedPL = 0

            else:
                # first action
                if action == 0:
                    # buy for the first time
                    self.buy_price = current_price
                    self.position_size = 0.2 * self.balance * 100
                elif action == 1:
                    # sell for the first time
                    self.sell_price = current_price
                    self.position_size = 0.2 * self.balance * 100

        else:
            # continue in same position
            if self.previous_action == 0:
                # long position or buy
                self.unrealizedPL = (current_price - self.buy_price) * self.position_size
            elif self.previous_action == 1:
                # short position or buy
                self.unrealizedPL = (self.sell_price - current_price) * self.position_size
            # always reset the unrealizedPL because it is recalculted
            self.balance = self.nav + self.unrealizedPL
            self.unrealizedPL = 0
        if self.nav > self.max_nav:
            self.max_nav = self.nav

    def reset(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.nav = INITIAL_ACCOUNT_BALANCE
        self.max_nav = INITIAL_ACCOUNT_BALANCE
        self.realizedPL = 0
        self.unrealizedPL = 0
        self.buy_price = 0
        self.sell_price = 0
        self.position_size = 0
        self.previous_action = None
        self.current_step = random.randint(MAX_STEPS, len(self.df.loc[:, 'Open'].values) - MAX_STEPS)
        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.nav - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {profit}')
        print(f'Net worth: {self.nav} (Max net worth: {self.max_nav})')

    def _next_observation(self):
        # Get the data points for the last hour and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step - MAX_STEPS: self.current_step, 'Open'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - MAX_STEPS: self.current_step, 'High'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - MAX_STEPS: self.current_step, 'Low'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - MAX_STEPS: self.current_step, 'Close'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - MAX_STEPS: self.current_step, 'Volume'].values / MAX_VOLUME,
        ])

        # Append additional data and scale each value to between 0-1
        # obs = np.append(frame, [np.hstack((
        #     np.array([
        #         self.balance / MAX_ACCOUNT_BALANCE,
        #         self.max_nav / MAX_ACCOUNT_BALANCE,
        #         self.nav / MAX_ACCOUNT_BALANCE,
        #         self.unrealizedPL / MAX_ACCOUNT_BALANCE,
        #         self.realizedPL / MAX_ACCOUNT_BALANCE,
        #         self.position_size / MAX_ACCOUNT_BALANCE * 100,
        #         self.buy_price / MAX_CURRENCY_PRICE
        #     ]),
        #     np.zeros(MAX_STEPS - 6))
        # )], axis=0)
        return frame.ravel()