import os, time, requests, json, random

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from spapi import SPAPIConfig, SPAPIClient, ApiException#, OrdersV0Api

from spapi.rest import ApiException

load_dotenv()

# ---- Invoice Config ----
INVOICE_POST_URL = os.getenv("INVOICE_POST_URL", "http://localhost:8000/invoices")

SELLER = {
    "company": os.getenv("SELLER_COMPANY", "").strip(),
    "abn": os.getenv("SELLER_ABN", "").strip(),
    "email": os.getenv("SELLER_EMAIL", "").strip(),
    "phone": os.getenv("SELLER_PHONE", "").strip(),
    "address": {
        "line1": os.getenv("SELLER_ADDRESS_LINE1", "").strip(),
        "line2": os.getenv("SELLER_ADDRESS_LINE2", "").strip(),
        "city": os.getenv("SELLER_CITY", "").strip(),
        "state": os.getenv("SELLER_STATE", "").strip(),
        "postcode": os.getenv("SELLER_POSTCODE", "").strip(),
        "country": os.getenv("SELLER_COUNTRY", "AU").strip(),
    },
    "logo_url": os.getenv("SELLER_LOGO_URL", "").strip(),
}


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
#orders_api = OrdersV0Api(sp_client.api_client)

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

def g(d, *keys, default=None):
    """Safe nested getter."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

def money(d, *path):
    v = g(d, *path)
    if isinstance(v, dict):
        return {"amount": g(v, "Amount", default=g(v, "amount")), "currency": g(v, "CurrencyCode", default=g(v, "currency"))}
    return {"amount": None, "currency": None}

def normalize_address(addr: dict | None) -> dict:
    if not addr:
        return {}
    return {
        "name": g(addr, "Name", default=g(addr, "name")),
        "line1": g(addr, "AddressLine1", default=g(addr, "address_line1")),
        "line2": g(addr, "AddressLine2", default=g(addr, "address_line2")),
        "line3": g(addr, "AddressLine3", default=g(addr, "address_line3")),
        "city": g(addr, "City", default=g(addr, "city")),
        "state": g(addr, "StateOrRegion", default=g(addr, "state_or_region")),
        "postcode": g(addr, "PostalCode", default=g(addr, "postal_code")),
        "country": g(addr, "CountryCode", default=g(addr, "country_code")),
        "phone": g(addr, "Phone", default=g(addr, "phone")),
    }

def build_invoice_payload(order: dict, items: list[dict], buyer_payload: dict | None, address_payload: dict | None) -> dict:
    order_id = g(order, "AmazonOrderId", default=g(order, "amazon_order_id"))
    buyer_name = g(buyer_payload or {}, "BuyerInfo", "BuyerName", default=g(buyer_payload or {}, "buyer_info", "buyer_name"))
    buyer_email = g(buyer_payload or {}, "BuyerInfo", "BuyerEmail", default=g(buyer_payload or {}, "buyer_info", "buyer_email"))
    ship_addr = normalize_address(g(address_payload or {}, "ShippingAddress", default=g(address_payload or {}, "shipping_address")))
    purchase_date = g(order, "PurchaseDate", default=g(order, "purchase_date"))
    currency = g(order, "OrderTotal", "CurrencyCode", default=g(order, "order_total", "currency"))

    # Line items
    line_items = []
    subtotal = 0.0
    tax_total = 0.0
    shipping_total = 0.0
    discount_total = 0.0

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    for it in items:
        qty = g(it, "QuantityOrdered", default=g(it, "quantity_ordered", default=1))
        title = g(it, "Title", default=g(it, "title"))
        asin = g(it, "ASIN", default=g(it, "asin"))
        sku = g(it, "SellerSKU", default=g(it, "seller_sku"))
        unit_price = money(it, "ItemPrice")
        item_tax = money(it, "ItemTax")
        ship_price = money(it, "ShippingPrice")
        ship_tax = money(it, "ShippingTax")
        promo_disc = money(it, "PromotionDiscount")
        gift_wrap_price = money(it, "GiftWrapPrice")
        gift_wrap_tax = money(it, "GiftWrapTax")

        # Note: SP-API's ItemPrice is typically the total **for the line**, not per-unit.
        line_subtotal = to_float(unit_price["amount"])
        line_tax = to_float(item_tax["amount"]) + to_float(gift_wrap_tax["amount"])
        line_shipping = to_float(ship_price["amount"]) + to_float(ship_tax["amount"])
        line_discount = abs(to_float(promo_disc["amount"]))

        subtotal += line_subtotal
        tax_total += line_tax
        shipping_total += line_shipping
        discount_total += line_discount

        line_items.append({
            "title": title,
            "asin": asin,
            "sku": sku,
            "quantity": qty,
            "prices": {
                "item_price": unit_price,
                "item_tax": item_tax,
                "shipping_price": ship_price,
                "shipping_tax": ship_tax,
                "promotion_discount": promo_disc,
                "gift_wrap_price": gift_wrap_price,
                "gift_wrap_tax": gift_wrap_tax,
            },
            "line_totals": {
                "subtotal": {"amount": line_subtotal, "currency": currency},
                "tax": {"amount": line_tax, "currency": currency},
                "shipping": {"amount": line_shipping, "currency": currency},
                "discount": {"amount": line_discount, "currency": currency},
            }
        })

    # Prefer Amazon's grand total if present
    order_total = money(order, "OrderTotal")
    grand_total = order_total["amount"]
    if grand_total is None:
        grand_total = round(subtotal + tax_total + shipping_total - discount_total, 2)

    payload = {
        "invoice_id": order_id,               # you can change to your own numbering
        "currency": currency or "AUD",
        "order": {
            "amazon_order_id": order_id,
            "purchase_date": purchase_date,
            "order_status": g(order, "OrderStatus", default=g(order, "order_status")),
            "fulfillment_channel": g(order, "FulfillmentChannel", default=g(order, "fulfillment_channel")),
            "sales_channel": g(order, "SalesChannel", default=g(order, "sales_channel")),
            "marketplace_id": g(order, "MarketplaceId", default=g(order, "marketplace_id")),
        },
        "buyer": {"name": buyer_name, "email": buyer_email},
        "shipping_address": ship_addr,
        "items": line_items,
        "totals": {
            "items_subtotal": {"amount": round(subtotal, 2), "currency": currency},
            "tax": {"amount": round(tax_total, 2), "currency": currency},
            "shipping": {"amount": round(shipping_total, 2), "currency": currency},
            "discounts": {"amount": round(discount_total, 2), "currency": currency},
            "grand_total": {"amount": to_float(grand_total), "currency": currency},
        },
        "seller": SELLER,
        "notes": "Thank you for your order.",
        "generated_at": utcnow_iso(),
    }
    return payload

def post_to_invoice_service(payload: dict) -> dict:
    r = requests.post(INVOICE_POST_URL, json=payload, timeout=30)
    try:
        data = r.json()
    except Exception:
        data = {"text": r.text}
    return {"status_code": r.status_code, "response": data}

def build_client_for_region(region: str) -> SPAPIClient:
    cfg = SPAPIConfig(
        client_id=LWA_CLIENT_ID,
        client_secret=LWA_CLIENT_SECRET,
        refresh_token=LWA_REFRESH_TOKEN,
        region=region,   # "FE" (AU), "NA", or "EU"
    )
    return SPAPIClient(cfg)

def raw_call_with_client(api_client, resource_path, method="GET", query=None, headers=None, max_attempts=8):
    import json, time, random
    from spapi.rest import ApiException

    query = query or []
    headers = headers or {}
    delay = 2.0
    for _ in range(max_attempts):
        try:
            resp = api_client.call_api(
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
            if getattr(e, "status", None) in (429, 500, 502, 503, 504):
                retry_after = None
                try:
                    retry_after = float(getattr(e, "headers", {}).get("Retry-After", 0))
                except Exception:
                    pass
                sleep_for = max(retry_after or 0, delay) + random.random()
                time.sleep(sleep_for)
                delay = min(delay * 2, 90)
                continue
            raise


# --- Tokens (RDT) via LWA + raw (same as before but kept here for completeness) ---
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


REGIONS = ["FE", "NA", "EU"]  # FE = AU/JP/SG

def fetch_order_one(order_id: str, region: str):
    client = build_client_for_region(region)
    return raw_call_with_client(client.api_client, f"/orders/v0/orders/{order_id}", "GET")

def find_order_across_regions(order_id: str, first_region: str = "FE"):
    regions = [first_region] + [r for r in REGIONS if r != first_region]
    last_err = None
    for r in regions:
        try:
            data = fetch_order_one(order_id, r)
            payload = data.get("payload", {}) if "payload" in data else data
            orders = payload.get("Orders", []) or payload.get("orders", [])
            if orders:
                return {"region": r, "order": orders[0]}
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("Order not found in any region")



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
    include_pii: bool = Query(False),
):
    # 1) Fetch by ID using getOrders (not getOrder) to avoid 404 weirdness
    data = raw_call("/orders/v0/orders", "GET", query=[("AmazonOrderIds", order_id)])
    payload = data.get("payload", {}) if "payload" in data else data
    orders = payload.get("Orders", []) or payload.get("orders", [])
    if not orders:
        raise HTTPException(404, detail=f"Order {order_id} not found")
    order = orders[0]

    # 2) Items
    if include_items:
        order["items"] = raw_get_order_items(order_id)

    # 3) (Optional) PII via RDT if you have the roles
    if include_pii:
        try:
            pii = fetch_pii_for_order_raw(order_id, want_buyer=True, want_address=True, want_tax=True)
            if pii.get("buyer"):
                order["buyer"] = pii["buyer"].get("BuyerInfo") or pii["buyer"]
            if pii.get("address"):
                order["shippingAddress"] = pii["address"].get("ShippingAddress") or pii["address"]
        except Exception as e:
            order["_pii_error"] = {"message": str(e)}

    return order

@app.get("/orders/{order_id}/items")
def get_order_items(order_id: str):
    try:
        items = fetch_order_items(order_id)
        return {"orderId": order_id, "count": len(items), "items": items}
    except ApiException as e:
        raise HTTPException(status_code=e.status or 500, detail=e.body or str(e))


@app.post("/orders/{order_id}/invoice")
def generate_invoice_for_order(
    order_id: str,
    post: bool = Query(True, description="POST to invoice service after building payload"),
    include_pii: bool = Query(True, description="Buyer name/email + shipping address via RDT"),
    region: str = Query("FE", description="FE (AU/JP/SG), NA, or EU"),
    try_all_regions: bool = Query(True, description="Try other regions if not found in `region`"),
):
    # --- 1) HEADER: use getOrders?AmazonOrderIds={id} (raw) and optionally try all regions ---
    search_order = [region] + [r for r in ["FE", "NA", "EU"] if r != region] if try_all_regions else [region]

    found = None
    used_region = None
    for r in search_order:
        client = build_client_for_region(r)
        try:
            resp = raw_call_with_client(
                client.api_client,
                "/orders/v0/orders",
                "GET",
                query=[("AmazonOrderIds", order_id)],
            )
            payload = resp.get("payload", {}) if "payload" in resp else resp
            orders = payload.get("Orders", []) or payload.get("orders", [])
            if orders:
                found = orders[0]
                used_region = r
                break
        except Exception:
            # keep looping through regions
            continue

    if not found:
        raise HTTPException(404, detail=f"Order {order_id} not found")

    order = found
    order_id_norm = order.get("AmazonOrderId") or order.get("amazon_order_id")

    # --- 2) ITEMS: use the same region we found the order in ---
    client = build_client_for_region(used_region)
    items = []
    next_token = None
    while True:
        query = [("NextToken", next_token)] if next_token else None
        data = raw_call_with_client(
            client.api_client,
            f"/orders/v0/orders/{order_id_norm}/orderItems",
            "GET",
            query=query,
        )
        payload = data.get("payload", {}) if "payload" in data else data
        batch = payload.get("OrderItems", []) or payload.get("order_items", [])
        items.extend(batch)
        next_token = payload.get("NextToken") or payload.get("next_token")
        if not next_token:
            break

    # --- 3) PII (if roles granted): RDT against the same region client ---
    buyer_payload = address_payload = None
    if include_pii:
        try:
            def fetch_pii_with_client(order_id_inner: str, api_client):
                # LWA access token
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
                access_token = r.json()["access_token"]

                REGION_HOSTS = {
                    "NA": "sellingpartnerapi-na.amazon.com",
                    "EU": "sellingpartnerapi-eu.amazon.com",
                    "FE": "sellingpartnerapi-fe.amazon.com",
                }
                host = api_client.configuration.host
                region_code = next((k for k, v in REGION_HOSTS.items() if v in host), "FE")

                body = {
                    "restrictedResources": [
                        {
                            "method": "GET",
                            "path": f"/orders/v0/orders/{order_id_inner}/buyerInfo",
                            "dataElements": ["buyerInfo", "buyerTaxInformation"],
                        },
                        {
                            "method": "GET",
                            "path": f"/orders/v0/orders/{order_id_inner}/address",
                            "dataElements": ["shippingAddress"],
                        },
                    ]
                }
                rdt_resp = requests.post(
                    f"https://{REGION_HOSTS[region_code]}/tokens/2021-03-01/restrictedDataToken",
                    headers={"content-type": "application/json", "x-amz-access-token": access_token},
                    json=body,
                    timeout=30,
                )
                rdt_resp.raise_for_status()
                rdt = rdt_resp.json()["restrictedDataToken"]

                # Swap token to RDT for the restricted calls
                cfg = api_client.configuration
                original = cfg.access_token
                cfg.access_token = rdt
                try:
                    buyer = raw_call_with_client(api_client, f"/orders/v0/orders/{order_id_inner}/buyerInfo", "GET").get("payload")
                    address = raw_call_with_client(api_client, f"/orders/v0/orders/{order_id_inner}/address", "GET").get("payload")
                    return {"buyer": buyer, "address": address}
                finally:
                    cfg.access_token = original

            pii = fetch_pii_with_client(order_id_norm, client.api_client)
            buyer_payload = pii.get("buyer")
            address_payload = pii.get("address")
        except Exception:
            # Proceed without PII if roles/token are not available
            buyer_payload = address_payload = None

    # --- 4) Assemble invoice payload + optional POST to your tool ---
    payload = build_invoice_payload(order, items, buyer_payload, address_payload)
    result = post_to_invoice_service(payload) if post else None

    return {
        "region_used": used_region,
        "posted": bool(post),
        "invoice_post_url": INVOICE_POST_URL if post else None,
        "post_result": result,
        "payload": payload,  # raw JSON for your generator
    }

# Optional: enable `python main.py` to run locally
if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI on port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
