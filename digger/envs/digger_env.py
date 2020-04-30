import random
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

INITIAL_ACCOUNT_BALANCE = 20000.0
MAX_ACCOUNT_BALANCE = 2000000.0
MAX_STEPS = 1000
MAX_CURRENCY_PRICE = 500
MAX_VOLUME = 1000


class DiggerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df=None):
        self.df = df
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.nav = INITIAL_ACCOUNT_BALANCE
        self.max_nav = INITIAL_ACCOUNT_BALANCE
        self.realizedPL = 0
        self.unrealizedPL = 0
        self.trades = 0
        self.buy_price = 0
        self.position_size = 0
        # observation space - ohlc of the past 12 ticks (an hour)
        # action space - buy at most 20% of asset net asset, with a take profit of 60% or wait
        # self.observation_space =
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=((MAX_STEPS + 1) * 6,))

        self.current_step = random.randint(MAX_STEPS, len(self.df.loc[:, 'Open'].values) - MAX_STEPS)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'Open'].values) - MAX_STEPS:
            self.current_step = MAX_STEPS
        delay_modifier = ((self.current_step - MAX_STEPS)/ 6000000000)

        reward = (self.balance + self.unrealizedPL) * delay_modifier
        # reward = self.balance * delay_modifier
        done = self.nav <= 0
        obs = self._next_observation()
        info = {
            'current_step': self.current_step,
            'delay_modifier': delay_modifier,
            'reward': reward,
            'balance': self.balance,
            'position_size': self.position_size,
            'nav': self.nav,
            'unrealizedPL': self.unrealizedPL,
            'buy_price': self.buy_price
        }
        return obs, reward, done, info

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"],
            self.df.loc[self.current_step, "Close"])

        self.unrealizedPL = (current_price - self.buy_price) * self.position_size
        # take profit to stop loss ratio is 3:1
        if self.trades == 0:
            # no trades
            self.trades = 1
            if action == 0:
                # buy
                self.buy_price = current_price
                self.position_size = 0.05 * self.balance * 100
            elif action == 1:
                # hold
                self.balance = self.balance * 0.999
            elif action == 2:
                self.sell_price = current_price
                self.position_size = 0.05 * self.balance * 100
        else:
            # theres an existing trade check for take profit and stop loss
            # take_profit_amount = 0.05 * self.balance * 1.3 # 1.6
            stop_loss_amount = 0.05 * self.balance * 0.75  # 20% stop loss
            take_profit_amount = 3 * stop_loss_amount

            if self.unrealizedPL >= take_profit_amount:
                self.trades = 0
                self.buy_price = 0
                self.position_size = 0
                self.balance += self.unrealizedPL
                self.realizedPL += self.unrealizedPL
                self.unrealizedPL = 0

            if self.unrealizedPL <= -stop_loss_amount:
                self.trades = 0
                self.buy_price = 0
                self.position_size = 0
                self.balance += self.unrealizedPL
                self.realizedPL += self.unrealizedPL
                self.unrealizedPL = 0

        self.nav += self.unrealizedPL
        if self.nav > self.max_nav:
            self.max_nav = self.nav

    def reset(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.nav = INITIAL_ACCOUNT_BALANCE
        self.max_nav = INITIAL_ACCOUNT_BALANCE
        self.realizedPL = 0
        self.unrealizedPL = 0
        self.trades = 0
        self.buy_price = 0
        self.position_size = 0
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
        obs = np.append(frame, [np.hstack((
            np.array([
                self.balance / MAX_ACCOUNT_BALANCE,
                self.max_nav / MAX_ACCOUNT_BALANCE,
                self.nav / MAX_ACCOUNT_BALANCE,
                self.unrealizedPL / MAX_ACCOUNT_BALANCE,
                self.realizedPL / MAX_ACCOUNT_BALANCE,
                self.trades,
                self.position_size / MAX_ACCOUNT_BALANCE * 100,
                self.buy_price / MAX_CURRENCY_PRICE
            ]),
            np.zeros(993))
        )], axis=0)
        return obs.ravel()
