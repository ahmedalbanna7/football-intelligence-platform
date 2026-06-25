# app/modules/players/service.py

from sqlalchemy.orm import Session
from app.models.player import Player


def create_player(
    db: Session,
    name: str,
    age: int,
    position: str
):
    player = Player(
        name=name,
        age=age,
        position=position
    )

    db.add(player)
    db.commit()
    db.refresh(player)

    return player