# Supported environment names, grouped by backend. `env_wrapper.Env` dispatches on these lists.

# pip install gymnasium[classic-control]
GYM_ENVS_CLASSIC = [
    "Pendulum-v1",
    "MountainCarContinuous-v0",
]

# pip install gymnasium[box2d]  (also requires: pip install swig)
GYM_ENVS_BOX2D = [
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "CarRacing-v3",
]

# pip install gymnasium[mujoco]
GYM_ENVS_MUJOCO = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "InvertedDoublePendulum-v5",
    "InvertedPendulum-v5",
    "Pusher-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]

GYM_ENVS = GYM_ENVS_CLASSIC + GYM_ENVS_BOX2D + GYM_ENVS_MUJOCO


DMCONTROL_ENVS = [
    "cartpole-balance",
    "cartpole-balance-sparse",
    "cartpole-swingup",
    "cartpole-swingup-sparse",
    "finger-spin",
    "finger-turn-easy",
    "finger-turn-hard",
    "cheetah-run",
    "reacher-easy",
    "reacher-hard",
    "cup-catch",
    "walker-stand",
    "walker-walk",
    "walker-run",
    "hopper-stand",
    "hopper-hop",
    "humanoid-stand",
    "humanoid-walk",
    "humanoid-run",
]


# pip install ale-py  (ROMs included). The 55-game DreamerV2 benchmark: the standard 57-game ALE suite
# without Defender and Surround.
ATARI_GAMES = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "DoubleDunk",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
]

ATARI_ENVS = [f"ALE/{game}-v5" for game in ATARI_GAMES]


# Environments that supports early termination
TERMINATING_ENVS = {
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "CarRacing-v3",
    "MountainCarContinuous-v0",
    *ATARI_ENVS,  # episodes end on game over
}
