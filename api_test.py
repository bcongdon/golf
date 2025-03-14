from pettingzoo.test import api_test
from golf_game_env_v2 import env


for i in range(1000):
    api_test(env(render_mode="human"), num_cycles=1000)