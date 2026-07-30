import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text

from app.api import router
from app.config import ROOT, settings
from app.database import Base, SessionLocal, engine
from app.models import Camera, TelegramSubscriber
from app.services.events import event_hub
from app.services.telegram import poll_telegram_updates


def _ensure_sqlite_compatibility():
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(incidents)")).fetchall()
        }
        if "incident_metadata" not in columns:
            connection.execute(
                text("ALTER TABLE incidents ADD COLUMN incident_metadata JSON DEFAULT '{}'")
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_compatibility()
    db = SessionLocal()
    try:
        camera = db.get(Camera, "building-a-exit-south")
        if camera is None and not db.scalar(select(Camera).limit(1)):
            db.add(
                Camera(
                    id="building-a-exit-south",
                    name="South Access Camera",
                    facility="Building A",
                    zone="South Access Zone",
                    exit_zone=[[250, 100], [850, 100], [850, 680], [250, 680]],
                    blocked_classes=sorted(settings.blocked_class_set),
                    confidence_threshold=settings.detection_threshold,
                    minimum_overlap=settings.default_minimum_overlap,
                    persistence_seconds=settings.minimum_duration_seconds,
                    alert_cooldown_seconds=settings.alert_cooldown_seconds,
                )
            )
            db.commit()
        elif camera is not None:
            if camera.name == "South Exit Camera":
                camera.name = "South Access Camera"
            if camera.zone == "Fire Exit South":
                camera.zone = "South Access Zone"
            db.commit()
        if settings.user_id:
            subscriber = db.get(TelegramSubscriber, settings.user_id)
            if subscriber is None:
                db.add(
                    TelegramSubscriber(
                        chat_id=settings.user_id,
                        username="configured-user",
                        active=True,
                    )
                )
            else:
                subscriber.active = True
            db.commit()
    finally:
        db.close()
    polling_task = asyncio.create_task(poll_telegram_updates())
    try:
        yield
    finally:
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Local-first facility safety monitoring and incident orchestration.",
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
app.mount("/evidence", StaticFiles(directory=ROOT / "evidence"), name="evidence")


@app.websocket("/api/events")
async def websocket_events(websocket: WebSocket):
    await event_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_hub.disconnect(websocket)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "app" / "static" / "index.html")
