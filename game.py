from enum import Enum
from itertools import product
from typing import List

import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces

from configs import BaseConfig


class Action(Enum):
    LEFT = 0
    RIGHT = 1


class Player:
    def __init__(self, id, x, y, speed, angular_speed):
        self.id = id
        self.color = pg.Color(
            np.random.randint(low=0, high=255),
            np.random.randint(low=0, high=255),
            np.random.randint(low=0, high=255),
        )
        self.x = x
        self.y = y
        self.direction = np.random.rand() * 2 * np.pi
        self.speed = speed
        self.angular_speed = angular_speed
        self.alive = True

    def move(self, action):
        if not self.alive:
            return

        print(f"Player {self.id}: ({self.x}, {self.y}) -> {self.direction}")

        if action == Action.LEFT:
            self.direction -= self.angular_speed
        elif action == Action.RIGHT:
            self.direction += self.angular_speed
        self.x += self.speed * np.cos(self.direction)
        self.y += self.speed * np.sin(self.direction)

    def to_grid(self):
        return int(self.x), int(self.y)


class KurveEnv(gym.Env):
    def __init__(
        self,
        n_players: int,
        board_size: List[int],
        speed: float,
        angular_speed: float,
        scale: int,
        fps: int,
        render_mode: str = "human",
    ):
        super(KurveEnv, self).__init__()

        self.players = []
        self.board_size = board_size
        self.n_players = n_players
        self.speed = speed
        self.angular_speed = angular_speed
        self.render_mode = render_mode
        self.scale = scale
        self.fps = fps

        self.window = None
        self.clock = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=self.n_players, shape=board_size, dtype=np.int32
        )

    def step(self, actions):
        assert len(actions) == len(self.players), "Actions missmatch"

        for p, a in zip(self.players, actions):
            print(f"Player {p.id} is alive: {p.alive}")
            p.move(a)

        for p in self.players:
            x, y = p.to_grid()
            if self.board[y, x] not in (0, p.id):
                p.alive = False
            else:
                self.board[y, x] = p.id

        reward = 0
        terminated = not any([p.alive for p in self.players])
        return self.board, reward, terminated, {}

    def reset(self):
        self.players = []
        initial_pos = np.random.randint(0, self.board_size[0], size=(self.n_players, 2))
        colors = [pg.Color("red"), pg.Color("green")]
        for i in range(self.n_players):
            x, y = initial_pos[i]
            p = Player(i + 1, x, y, self.speed, self.angular_speed)
            p.color = colors[i]
            self.players.append(p)

        self.board = np.zeros(self.board_size, dtype=np.int32)
        self.board[0, :] = -1
        self.board[-1, :] = -1
        self.board[:, 0] = -1
        self.board[:, -1] = -1
        for p in self.players:
            self.board[p.y, p.x] = p.id
        self.colors = [p.color for p in self.players]

        return self.board

    def _render_frame(self):
        pg.init()
        pg.display.set_caption("Kurve")
        window_size = (self.board_size[0] * self.scale, self.board_size[1] * self.scale)
        if self.window is None:
            self.window = pg.display.set_mode(window_size)
        if self.clock is None:
            self.clock = pg.time.Clock()

        canvas = pg.Surface(window_size)
        canvas.fill(pg.Color("black"))

        rows = range(self.board_size[0])
        cols = range(self.board_size[1])
        for x, y in product(rows, cols):
            pixel = self.board[y, x]
            border = (x * self.scale, y * self.scale, self.scale, self.scale)
            if pixel > 0:
                pg.draw.rect(canvas, self.colors[pixel - 1], border)
            elif pixel == -1:
                pg.draw.rect(canvas, pg.Color("yellow"), border)

        self.window.blit(canvas, canvas.get_rect())
        pg.event.pump()
        pg.display.update()
        self.clock.tick(self.fps)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()


def main():
    config = BaseConfig()

    env = KurveEnv(
        n_players=config.n_players,
        board_size=config.board_size,
        speed=config.speed,
        angular_speed=config.angular_speed,
        render_mode="human",
        scale=config.scale,
        fps=config.fps,
    )

    observation = env.reset()

    for _ in range(20):
        # action = env.action_space.sample()
        actions = [Action.LEFT, Action.LEFT]
        observation, reward, terminated, info = env.step(actions)
        env.render()

        if terminated:
            observation = env.reset()

    env.close()


if __name__ == "__main__":
    main()
