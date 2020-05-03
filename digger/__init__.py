from gym.envs.registration import register

register(
    id='DiggerEnv-v0',
    entry_point='digger.envs:DiggerEnv',
)

register(
    id='StrategyEnv-v0',
    entry_point='digger.envs:StrategyEnv',
)

register(
    id='BSEnv-v0',
    entry_point='digger.envs:BSEnv',
)
