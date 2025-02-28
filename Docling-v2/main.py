from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.routesV2 import router as routes_router  # Correctly import API routes

app = FastAPI()

# Mount API routes
app.include_router(routes_router, prefix="/api/v2")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("index.html")
