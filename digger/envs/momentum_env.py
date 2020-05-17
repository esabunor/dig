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


class MomentumEnv(gym.Env):
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
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, (self.max_steps + 1)))

        self.current_step = 9 #random.randint(self.max_steps, len(self.df.loc[:, 'Open'].values) - self.max_steps)
        self.initial_step = self.current_step
        self.take_profit = 0
        self.stop_loss = 0
        self.action = None
        self.open_position = False

    def step(self, action):
        # Execute one time step within the environment
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "Close"]
        reward, price_diff = self._take_action(action, current_price)

        if self.current_step > len(self.df.loc[:, 'Close'].values) - 9:
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
            'current_postn': self.action
        }
        self.current_step += 1

        return obs, reward, done, info

    def _take_action(self, action, current_price):
        # take profit to stop loss ratio is 3:1
        reward, price_diff = 0, 0
        if not self.open_position:
            if action == 1:
                # buy
                # previous action is sell so calculate profit for the short position
                # long
                price_diff = self.sell_price - current_price
                reward = price_diff * 1000
                self.unrealizedPL = (self.sell_price - current_price) * self.position_size
                self.buy_price = current_price
                self.position_size = 0.2 * self.balance * 100
                self.take_profit = current_price + 0.002
                self.stop_loss = current_price - 0.0007
                self.action = action
                self.open_position = True
            elif action == 2:
                # sell
                # short|
                price_diff = current_price - self.buy_price
                reward = price_diff * 1000
                self.unrealizedPL = (current_price - self.buy_price) * self.position_size  # calculate profit for previous postion
                self.sell_price = current_price
                self.position_size = 0.2 * self.balance * 100
                self.take_profit = current_price - 0.002
                self.stop_loss = current_price + 0.0007
                self.action = action
                self.open_position = True
            elif action == 0:
                pass  # do nothing

        if self.action == 1:
            # buy
            if self.take_profit >= current_price or self.stop_loss <= current_price:
                # take profit or losss
                self.balance = self.nav + self.unrealizedPL
                self.realizedPL += self.unrealizedPL
                self.nav = self.balance
                self.open_position = False

        elif self.action == 2:
            # sell
            if self.take_profit <= current_price or self.stop_loss >= current_price:
                # take profit
                self.balance = self.nav + self.unrealizedPL
                self.realizedPL += self.unrealizedPL
                self.nav = self.balance
                self.open_position = False
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
        self.take_profit = 0
        self.stop_loss = 0
        self.action = None
        self.open_position = False
        self.current_step =  9 #random.randint(9, len(self.df.loc[:, 'Close'].values) - 9)
        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.nav - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {profit}')
        print(f'Net worth: {self.nav} (Max net worth: {self.max_nav})')

    def _next_observation(self):
        # Get the data points for the last hour and scale to between 0-1

        n_steps = 9
        t0 = np.array(self.current_step).reshape(-1, 1) - 9
        ys_cp = self.df.Close.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_op = self.df.Open.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_hp = self.df.High.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_lp = self.df.Low.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_gx = self.df.Green_X.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_rx = self.df.Red_X.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_volume = self.df.Volume.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        ys_macd = self.df.MACD.values[t0.astype('int') + np.arange(0, n_steps + 1)]
        return np.c_[ys_cp,
                       ys_op,
                       ys_hp,
                       ys_lp,
                       ys_volume,
                       ys_macd,
                       ys_gx,
                       ys_rx]