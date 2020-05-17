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

register(
    id='BSEnv-v1',
    entry_point='digger.envs:BSEnvV1',
)

register(
    id='BSEnv-v2',
    entry_point='digger.envs:BSEnvV2',
)
register(
    id='BSEnvTester-v0',
    entry_point='digger.envs:BSEnvTester',
)

register(
    id='MomentumEnv-v0',
    entry_point='digger.envs:MomentumEnv',
)
