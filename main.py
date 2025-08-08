import os, time, requests, json, random

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from spapi import SPAPIConfig, SPAPIClient, ApiException, OrdersV0Api

from spapi.rest import ApiException

load_dotenv()

# ---- Config ----
REGION = os.getenv("SPAPI_REGION", "FE")  # AU is in FE
MARKETPLACE_ID = os.getenv("SPAPI_MARKETPLACE_ID", "A39IBJ37TRP1C6")
PORT = int(os.getenv("PORT", "5000"))

LWA_CLIENT_ID = os.getenv("LWA_CLIENT_ID")
LWA_CLIENT_SECRET = os.getenv("LWA_CLIENT_SECRET")
LWA_REFRESH_TOKEN = os.getenv("LWA_REFRESH_TOKEN")

if not all([LWA_CLIENT_ID, LWA_CLIENT_SECRET, LWA_REFRESH_TOKEN]):
    raise RuntimeError("Missing LWA env vars. Check .env.example.")

# Build a shared SP-API client (SDK handles token exchange & auth)
sp_config = SPAPIConfig(
    client_id=LWA_CLIENT_ID,
    client_secret=LWA_CLIENT_SECRET,
    refresh_token=LWA_REFRESH_TOKEN,
    region=REGION,   # "FE" for AU
    scope=None       # optional; not needed for seller-auth calls
)

sp_client = SPAPIClient(sp_config)
orders_api = OrdersV0Api(sp_client.api_client)

app = FastAPI(title="Amazon SP-API Orders for AU", version="1.0.0")

@app.get("/__routes")
def __routes():
    return [r.path for r in app.routes]


# ---------- Utilities ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_iso8601(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def backoff_call(fn, *args, max_attempts=6, **kwargs):
    """Exponential backoff for rate limits (429) & transient errors (>=500)."""
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except ApiException as e:
            status = getattr(e, "status", None)
            # Retry on 429 + 5xx
            if status in (429, 500, 502, 503, 504):
                last_exc = e
                time.sleep(delay)
                delay *= 2  # exponential
                continue
            raise
    # Exhausted
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown failure without ApiException")

REGION_HOSTS = {
    "NA": "sellingpartnerapi-na.amazon.com",
    "EU": "sellingpartnerapi-eu.amazon.com",
    "FE": "sellingpartnerapi-fe.amazon.com",  # AU lives here
}

def raw_call(resource_path, method="GET", query=None, headers=None, max_attempts=8):
    query = query or []
    headers = headers or {}

    delay = 2.0  # start small, back off on 429/5xx
    for attempt in range(1, max_attempts + 1):
        try:
            resp = sp_client.api_client.call_api(
                resource_path, method,
                path_params={}, query_params=query,
                header_params=headers, body=None, post_params=[], files={},
                response_type=None, auth_settings=['bearerAuth'],
                _return_http_data_only=True, _preload_content=False
            )
            data = getattr(resp, "data", resp)
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            return json.loads(data)

        except ApiException as e:
            status = getattr(e, "status", None)
            # Retry on throttle or transient server errors
            if status in (429, 500, 502, 503, 504):
                # Honor Retry-After if present
                retry_after = None
                try:
                    retry_after = float(getattr(e, "headers", {}).get("Retry-After", 0))
                except Exception:
                    pass
                sleep_for = max(retry_after or 0, delay) + random.random()
                time.sleep(sleep_for)
                delay = min(delay * 2, 90)  # cap it so we don’t wait forever
                continue
            # Non-retryable -> bubble up
            raise

def raw_get_orders(
    marketplace_id: str,
    created_after: str | None,
    last_updated_after: str | None,
    order_statuses: list[str] | None,
    fulfillment_channels: list[str] | None,
    max_per_page: int,
    all_pages: bool,
    page_limit: int | None
):
    orders = []
    next_token = None
    pages = 0

    while True:
        if next_token:
            query = [("NextToken", next_token)]
        else:
            query = [("MarketplaceIds", marketplace_id)]
            if created_after:       query.append(("CreatedAfter", created_after))
            if last_updated_after:  query.append(("LastUpdatedAfter", last_updated_after))
            if order_statuses:
                for s in order_statuses:
                    query.append(("OrderStatuses", s))
            if fulfillment_channels:
                for c in fulfillment_channels:
                    query.append(("FulfillmentChannels", c))
            query.append(("MaxResultsPerPage", str(max_per_page)))

        data = raw_call("/orders/v0/orders", "GET", query=query)
        payload = data.get("payload", {}) if "payload" in data else data
        batch = payload.get("Orders", []) or payload.get("orders", [])
        orders.extend(batch)
        next_token = payload.get("NextToken") or payload.get("next_token")
        pages += 1

        # 🔑 small pacing delay between pages to avoid 429 bursts
        if next_token:
            time.sleep(3)  # adjust up if you still see 429s

        if not all_pages or not next_token:
            break
        if page_limit and pages >= page_limit:
            break

    return orders, next_token


def raw_get_order_items(order_id: str):
    items = []
    next_token = None
    while True:
        query = [("NextToken", next_token)] if next_token else None
        data = raw_call(f"/orders/v0/orders/{order_id}/orderItems", "GET", query=query)
        payload = data.get("payload", {}) if "payload" in data else data
        batch = payload.get("OrderItems", []) or payload.get("order_items", [])
        items.extend(batch)
        next_token = payload.get("NextToken") or payload.get("next_token")
        if not next_token:
            break
    return items


# --- Tokens (RDT) via LWA + raw (same as before but kept here for completeness) ---
import requests

def get_lwa_access_token() -> str:
    r = requests.post(
        "https://api.amazon.com/auth/o2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": LWA_REFRESH_TOKEN,
            "client_id": LWA_CLIENT_ID,
            "client_secret": LWA_CLIENT_SECRET,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def make_rdt_for_paths(paths, data_elements=None) -> str:
    base = f"https://{REGION_HOSTS[REGION]}"
    body = {
        "restrictedResources": [
            {"method": "GET", "path": p, **({"dataElements": data_elements} if data_elements else {})}
            for p in paths
        ]
    }
    at = get_lwa_access_token()
    r = requests.post(
        f"{base}/tokens/2021-03-01/restrictedDataToken",
        headers={"content-type": "application/json", "x-amz-access-token": at},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["restrictedDataToken"]

def fetch_pii_for_order_raw(order_id: str, want_buyer=True, want_address=True, want_tax=True):
    paths, elems = [], []
    if want_buyer:
        paths.append(f"/orders/v0/orders/{order_id}/buyerInfo")
        elems.append("buyerInfo")
        if want_tax:
            elems.append("buyerTaxInformation")
    if want_address:
        paths.append(f"/orders/v0/orders/{order_id}/address")
        elems.append("shippingAddress")

    rdt = make_rdt_for_paths(paths, data_elements=elems)

    # temporarily swap token in SDK so the raw call is signed using the RDT
    cfg = sp_client.api_client.configuration
    original = cfg.access_token
    cfg.access_token = rdt
    try:
        buyer = address = None
        if want_buyer:
            buyer = raw_call(f"/orders/v0/orders/{order_id}/buyerInfo").get("payload")
        if want_address:
            address = raw_call(f"/orders/v0/orders/{order_id}/address").get("payload")
        return {"buyer": buyer, "address": address}
    finally:
        cfg.access_token = original



class OrdersQuery(BaseModel):
    created_after: Optional[str] = None
    last_updated_after: Optional[str] = None
    order_statuses: Optional[List[str]] = None
    fulfillment_channels: Optional[List[str]] = None
    max_per_page: int = 100
    all_pages: bool = True
    page_limit: Optional[int] = None
    include_items: bool = True
    include_pii: bool = True


def merge_order_package(
    order: Dict[str, Any],
    items: Optional[List[Dict[str, Any]]] = None,
    buyer: Optional[Dict[str, Any]] = None,
    address: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(order)
    if items is not None:
        out["items"] = items
    if buyer is not None:
        out["buyer"] = buyer
    if address is not None:
        out["shippingAddress"] = address
    return out


def fetch_order_items(order_id: str) -> List[Dict[str, Any]]:
    collected = []
    next_token = None
    page = 0
    while True:
        if next_token:
            resp = backoff_call(orders_api.get_order_items, order_id, next_token=next_token)
        else:
            resp = backoff_call(orders_api.get_order_items, order_id)
        payload = resp.payload or {}
        items = payload.get("order_items") or payload.get("OrderItems") or []
        collected.extend(items)
        next_token = payload.get("next_token") or payload.get("NextToken")
        page += 1
        if not next_token:
            break
    return collected


def fetch_pii_for_order(order_id: str, want_buyer=True, want_address=True, want_tax=True):
    needed_paths = []
    if want_buyer:
        needed_paths.append(f"/orders/v0/orders/{order_id}/buyerInfo")
    if want_address:
        needed_paths.append(f"/orders/v0/orders/{order_id}/address")

    data_elements = []
    if want_buyer:
        data_elements.append("buyerInfo")
        if want_tax:
            data_elements.append("buyerTaxInformation")
    if want_address:
        data_elements.append("shippingAddress")

    rdt = make_rdt_for_paths(needed_paths, data_elements=data_elements)

    cfg = sp_client.api_client.configuration
    original_token = cfg.access_token
    cfg.access_token = rdt   # Use RDT as the access token for the restricted calls
    try:
        buyer = address = None
        if want_buyer:
            buyer_resp = backoff_call(orders_api.get_order_buyer_info, order_id)
            buyer = (buyer_resp.payload or {}).get("buyer_info") or buyer_resp.payload
        if want_address:
            addr_resp = backoff_call(orders_api.get_order_address, order_id)
            address = (addr_resp.payload or {}).get("shipping_address") or addr_resp.payload
        return {"buyer": buyer, "address": address}
    finally:
        cfg.access_token = original_token


# ---------- API Endpoints ----------

@app.get("/health")
def health():
    return {"ok": True, "time": utcnow_iso()}


@app.get("/orders")
def list_orders(
    created_after: str | None = Query(None),
    last_updated_after: str | None = Query(None),
    order_statuses: List[str] | None = Query(None),
    fulfillment_channels: List[str] | None = Query(None),
    max_per_page: int = Query(100, ge=1, le=100),
    all_pages: bool = Query(True),
    page_limit: int | None = Query(None, ge=1, le=1000),
    include_items: bool = Query(True),
    include_pii: bool = Query(False),   # default false until roles confirmed
):
    if not created_after and not last_updated_after:
        last_updated_after = to_iso8601(datetime.now(timezone.utc) - timedelta(days=7))

    orders, next_token = raw_get_orders(
        MARKETPLACE_ID, created_after, last_updated_after,
        order_statuses, fulfillment_channels, max_per_page,
        all_pages, page_limit
    )

    enriched = []
    for o in orders:
        order_id = o.get("AmazonOrderId") or o.get("amazon_order_id")
        merged = dict(o)

        if include_items and order_id:
            merged["items"] = raw_get_order_items(order_id)

        if include_pii and order_id:
            try:
                pii = fetch_pii_for_order_raw(order_id, want_buyer=True, want_address=True, want_tax=True)
                if pii.get("buyer"):
                    merged["buyer"] = pii["buyer"].get("BuyerInfo") or pii["buyer"]
                if pii.get("address"):
                    merged["shippingAddress"] = pii["address"].get("ShippingAddress") or pii["address"]
            except Exception as e:
                merged["_pii_error"] = {"message": str(e)}

        enriched.append(merged)

    return JSONResponse(
        content={
            "count": len(enriched),
            "orders": enriched,
            "next_token": next_token,
            "time": utcnow_iso(),
            "marketplace_id": MARKETPLACE_ID,
        }
    )


@app.get("/orders/{order_id}")
def get_order(
    order_id: str,
    include_items: bool = Query(True),
    include_pii: bool = Query(True),
):
    try:
        base = backoff_call(orders_api.get_order, order_id)
    except ApiException as e:
        raise HTTPException(status_code=e.status or 500, detail=e.body or str(e))

    order = (base.payload or {}).get("orders", [{}])
    order = order[0] if isinstance(order, list) and order else (base.payload or {})

    if include_items:
        items = fetch_order_items(order_id)
        order = merge_order_package(order, items=items)

    if include_pii:
        try:
            pii = fetch_pii_for_order(order_id, want_buyer=True, want_address=True, want_tax=True)
            order = merge_order_package(order, buyer=pii["buyer"], address=pii["address"])
        except ApiException as e:
            order["_pii_error"] = {
                "status": e.status,
                "message": "Restricted data unavailable. Check roles/RDT.",
            }

    return JSONResponse(content=order)


@app.get("/orders/{order_id}/items")
def get_order_items(order_id: str):
    try:
        items = fetch_order_items(order_id)
        return {"orderId": order_id, "count": len(items), "items": items}
    except ApiException as e:
        raise HTTPException(status_code=e.status or 500, detail=e.body or str(e))


# Optional: enable `python main.py` to run locally
if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI on port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
