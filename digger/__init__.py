from gym.envs.registration import register

register(
    id='DiggerEnv-v0',
    entry_point='digger.envs:DiggerEnv',
)