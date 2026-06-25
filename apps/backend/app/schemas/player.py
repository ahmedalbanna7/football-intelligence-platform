# app/schemas/player.py

from pydantic import BaseModel


class PlayerCreate(BaseModel):
    name: str
    age: int
    position: str


class PlayerResponse(PlayerCreate):
    id: int

    class Config:
        from_attributes = True