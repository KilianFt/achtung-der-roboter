from typing import List

from pydantic import BaseModel


class BaseConfig(BaseModel):
    speed: float = 5
    angular_speed: float = 0.2
    board_size: List[int] = [1000, 1000]
    n_players: int = 2
    scale: int = 1
    fps: int = 5
    player_radius: int = 10
