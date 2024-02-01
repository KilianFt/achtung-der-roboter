from typing import List

from pydantic import BaseModel


class BaseConfig(BaseModel):
    speed: float = 1
    angular_speed: float = 0.2
    board_size: List[int] = [100, 100]
    n_players: int = 2
    scale: int = 10
    fps: int = 5
