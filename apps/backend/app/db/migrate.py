import time

from alembic import command
from alembic.config import Config

MAX_RETRIES = 30
RETRY_DELAY_SECONDS = 2


def run_migrations() -> None:
    config = Config("alembic.ini")
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            command.upgrade(config, "head")
            print("Database migrations applied")
            return
        except Exception as exc:
            last_error = exc
            print(
                "Database is not ready for migrations yet "
                f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
            )
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError("Database migration retries exhausted") from last_error


if __name__ == "__main__":
    run_migrations()
