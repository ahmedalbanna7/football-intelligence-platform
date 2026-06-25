from fastapi import APIRouter

router = APIRouter()


@router.post("/")
def create_player():
    return {"message": "player created"}


@router.get("/")
def get_players():
    return {"message": "players"}