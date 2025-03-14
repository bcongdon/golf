from marllib import marl

# register pettingzoo environments
marl.register_env("golf_game_env_v2", golf_game_env_v2)

env = marl.make_env(environment_name="golf_game_env_v2", force_coop=True)
