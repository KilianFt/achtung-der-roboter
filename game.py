from typing import Tuple
from enum import Enum

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from configs import BaseConfig

class Action(Enum):
    LEFT = 0
    RIGHT = 1

class Player:
    def __init__(self, id, x, y, speed, angular_speed):
        self.id = id
        self.x = x
        self.y = y
        self.direction = np.random.rand() * 2 * np.pi
        self.speed = speed
        self.angular_speed = angular_speed
        self.alive = True

    def move(self, action):
        if not self.alive:
            return
        
        print(self.direction)
        print(self.x)
        print(self.y)

        if action == Action.LEFT:
            self.direction -= self.angular_speed
        elif action == Action.RIGHT:
            self.direction += self.angular_speed
        self.x += self.speed * np.cos(self.direction)
        self.y += self.speed * np.sin(self.direction)

    def to_grid(self):
        return int(self.x), int(self.y)


class KurveEnv(gym.Env):
    def __init__(self, n_players: int, board_size: Tuple[int], speed: float, angular_speed: float):
        super(KurveEnv, self).__init__()
        
        self.players = None
        self.board_size = board_size
        self.n_players = n_players
        self.speed = speed
        self.angular_speed = angular_speed

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=self.n_players, shape=board_size, dtype=int)

    def step(self, actions):
        assert len(actions) == len(self.players), 'Actions missmatch'

        for p, a in zip(self.players, actions):
            p.move(a)

        for p in self.players:
            x, y = p.to_grid()
            if self.board[y, x] != 0:
                p.alive = False
            else:
                self.board[y, x] = p.id

        reward = 0
        done = not any([p.alive for p in self.players])
        return self.board, reward, done, {}

    def reset(self):
        self.players = []
        initial_pos = np.random.randint(0, self.board_size[0], size=(self.n_players, 2))
        for i in range(self.n_players):
            x, y = initial_pos[i]
            p = Player(i, x, y, self.speed, self.angular_speed)
            self.players.append(p)

        self.board = np.zeros(self.board_size)
        for p in self.players:
            self.board[p.y, p.x] = p.id + 1

        return self.board

    def render(self, mode='human'):
        plt.imshow(self.board)
        plt.show()


def main():
    config = BaseConfig()

    env = KurveEnv(n_players=config.n_players, board_size=config.board_size, speed=config.speed,
                   angular_speed=config.angular_speed)

    observation = env.reset()

    for _ in range(1000):
        # action = env.action_space.sample()
        actions = [Action.LEFT, Action.LEFT]
        observation, reward, terminated, info = env.step(actions)
        env.render()

        if terminated:
            observation = env.reset()

    env.close()

if __name__ == '__main__':
    main()