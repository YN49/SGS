from gym.envs.registration import register

register(
    id='pic-v0',
    entry_point='pic.env:PIC'
)