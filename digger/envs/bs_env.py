import random
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from enum import Enum

INITIAL_ACCOUNT_BALANCE = 20000.0
MAX_ACCOUNT_BALANCE = 2000000.0
MAX_CURRENCY_PRICE = 500
MAX_VOLUME = 1000


class Positions(Enum):
    SHORT = 1
    LONG = 0


class Actions(Enum):
    SELL = 1
    BUY = 0


class BSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df=None, training=False, max_steps=100):
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
        self.max_steps = max_steps
        # observation space - ohlc of the past 12 ticks (an hour)
        # action space - buy at most 20% of asset net asset, with a take profit of 60% or wait
        # self.observation_space =
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, (self.max_steps + 1)))

        self.current_step = random.randint(self.max_steps, len(self.df.loc[:, 'Open'].values) - self.max_steps)
        self.initial_step = self.current_step
        self.previous_action = None  # buy

    def step(self, action):
        # Execute one time step within the environment
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "Close"]
        self.current_step += 1
        price_diff = 0

        reward, price_diff = self._take_action(action, current_price)

        if self.current_step > len(self.df.loc[:, 'Close'].values) - self.max_steps:
            self.current_step = self.max_steps

        # reward = (self.balance + self.unrealizedPL) * delay_modifier
        # reward = self.balance * delay_modifier
        done = self.nav * 0.9 > self.balance
        if done:
            reward *= 20
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
            'position': "Long" if action == 0 else "Short",
            'current_price': current_price,
            'price_diff': price_diff,
            'time': self.df.loc[self.current_step, "time"],
            'previous_postn': "Long" if self.previous_action == 0 else "Short"
        }
        self.previous_action = action
        return obs, reward, done, info

    def _take_action(self, action, current_price):
        # take profit to stop loss ratio is 3:1
        trade = False
        reward, price_diff = 0, 0
        if self.previous_action == None:
            trade = True
        else:
            if (action == 0 and self.previous_action == 1) or (action == 1 and self.previous_action == 0):
                trade = True

        if trade:
            if self.previous_action == 0 or self.previous_action == 1:
                # not the first action
                if action == 0:
                    # buy
                    # previous action is sell so calculate profit for the short position
                    # long
                    price_diff = self.sell_price - current_price
                    reward = price_diff * 1000
                    self.unrealizedPL = (self.sell_price - current_price) * self.position_size
                    self.buy_price = current_price
                    self.position_size = 0.2 * self.balance * 100
                elif action == 1:
                    # sell
                    # short|
                    price_diff = current_price - self.buy_price
                    reward = price_diff * 1000
                    self.unrealizedPL = (
                                                    current_price - self.buy_price) * self.position_size  # calculate profit for previous postion
                    self.sell_price = current_price
                    self.position_size = 0.2 * self.balance * 100
                self.balance = self.nav + self.unrealizedPL
                self.realizedPL += self.unrealizedPL
                self.nav = self.balance
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
                price_diff = current_price - self.buy_price
                reward += price_diff * 1000
            elif self.previous_action == 1:
                # short position or sell
                self.unrealizedPL = (self.sell_price - current_price) * self.position_size
                # short
                price_diff = self.sell_price - current_price
                reward += price_diff * 1000
            # always reset the unrealizedPL because it is recalculated
            self.balance = self.nav + self.unrealizedPL
        if self.nav > self.max_nav:
            self.max_nav = self.nav
        return reward, price_diff

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
        self.current_step = random.randint(self.max_steps, len(self.df.loc[:, 'Close'].values) - self.max_steps)
        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.nav - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {profit}')
        print(f'Net worth: {self.nav} (Max net worth: {self.max_nav})')

    def _next_observation(self):
        # Get the data points for the last hour and scale to between 0-1
        sma_5 = self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].rolling(window=5).mean()
        sma_8 = self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].rolling(window=8).mean()
        sma_13 = self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].rolling(window=13).mean()

        frame = np.array([
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Open'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'High'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Low'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Volume'].values / MAX_VOLUME,
            sma_5 / MAX_CURRENCY_PRICE,
            sma_8 / MAX_CURRENCY_PRICE,
            sma_13 / MAX_CURRENCY_PRICE
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
        #     np.zeros(self.max_steps - 6))
        # )], axis=0)
        return frame


class BSEnvV1(BSEnv):
    """
    includes moving averages 5, 8, 13
    observation shape is (808,) meaning that the input would look like
    """
    def __init__(self, df=None, training=False, max_steps=100):
        super().__init__(df, training)
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8 * (self.max_steps + 1),))

        self.current_step = random.randint(self.max_steps, len(self.df.loc[:, 'Open'].values) - self.max_steps)
        self.initial_step = self.current_step
        self.previous_action = None  # buy

    def _next_observation(self):
        # Get the data points for the last hour and scale to between 0-1
        sma_5 = self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].rolling(window=5).mean()
        sma_8 = self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].rolling(window=8).mean()
        sma_13 = self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].rolling(window=13).mean()

        frame = np.array([
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Open'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'High'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Low'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Volume'].values / MAX_VOLUME,
            sma_5 / MAX_CURRENCY_PRICE,
            sma_8 / MAX_CURRENCY_PRICE,
            sma_13 / MAX_CURRENCY_PRICE
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
        #     np.zeros(self.max_steps - 6))
        # )], axis=0)
        return frame.ravel()


class BSEnvV2(BSEnv):
    def __init__(self, df=None, training=False, max_steps=300):
        super().__init__(df, training)
        self.max_steps = max_steps # 300
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5 * (self.max_steps + 1),))

        self.current_step = random.randint(self.max_steps, len(self.df.loc[:, 'Open'].values) - self.max_steps)
        self.initial_step = self.current_step
        self.previous_action = None  # buy

    def _next_observation(self):
        # Get the data points for the last hour and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Open'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'High'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Low'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Close'].values / MAX_CURRENCY_PRICE,
            self.df.loc[self.current_step - self.max_steps: self.current_step, 'Volume'].values / MAX_VOLUME,
        ])
        return frame.ravel()