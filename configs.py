from typing import Tuple
from pydantic import BaseModel

class BaseConfig(BaseModel):
    speed: float = 1
    angular_speed: float = .2
    board_size: Tuple[int] = [100, 100]
    n_players: int = 2
