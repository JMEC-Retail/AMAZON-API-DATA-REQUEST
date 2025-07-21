import os
from sp_api.api import Orders
from sp_api.base import Marketplaces, SellingApiException
from dotenv import load_dotenv

load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
lwa_client_id = os.getenv("LWA_CLIENT_ID")
lwa_client_secret = os.getenv("LWA_CLIENT_SECRET")
lwa_refresh_token = os.getenv("LWA_REFRESH_TOKEN")
region = os.getenv("REGION", "ap-southeast-2")
marketplace_id = os.getenv("MARKETPLACE_ID", "A39IBJ37TRP1C6")


orders_client = Orders(
    credentials={
        "aws_access_key": aws_access_key,
        "aws_secret_key": aws_secret_key,
        "lwa_app_id": lwa_client_id,
        "lwa_client_secret": lwa_client_secret,
        "refresh_token": lwa_refresh_token,
    },
    marketplace=Marketplaces.AU
)


def get_recent_orders():
    try:
        response = orders_client.get_orders(
            MarketplaceIds=[marketplace_id],
            CreatedAfter="2024-07-01T00:00:00Z"
        )
        return response.payload.get('Orders', [])
    except SellingApiException as ex:
        raise Exception(f"Amazon SP-API Error: {ex}")
