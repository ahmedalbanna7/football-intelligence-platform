from minio import Minio
from app.core.config import settings

client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False
)

BUCKET_NAME = "videos"


def create_bucket_if_not_exists():

    if not client.bucket_exists(BUCKET_NAME):

        client.make_bucket(BUCKET_NAME)

        print(f"Bucket {BUCKET_NAME} created")

    else:

        print(f"Bucket {BUCKET_NAME} already exists")