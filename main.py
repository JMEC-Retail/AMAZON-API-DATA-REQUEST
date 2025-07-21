from fastapi import FastAPI, HTTPException
from amazon_orders import get_recent_orders

app = FastAPI()

@app.get("/orders")
async def get_orders():
    try:
        orders = get_recent_orders()
        return {"success": True, "orders": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
