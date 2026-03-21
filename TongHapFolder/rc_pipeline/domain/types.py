from dataclasses import dataclass
from typing import Optional


@dataclass
class FinalCommand:
    class_id: int
    angle: Optional[float]
    action: int
    status: str = ""
