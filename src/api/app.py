"""FastAPI Application Factory and Setup."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router as api_router
from src.config import settings
from src.model.inference import SpoilDetectionEngine
from src.utils.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "web"
SAMPLES_DIR = BASE_DIR.parent / "samples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager: warms up model on boot, cleans up on shutdown."""
    logger.info("Initializing %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    try:
        engine = SpoilDetectionEngine()
        logger.info("Inference engine ready. Loaded model with %d parameters.", engine.model.count_params() if engine.model else 0)
    except Exception as exc:
        logger.error("Failed to initialize model during startup: %s", str(exc), exc_info=True)

    yield

    logger.info("Shutting down %s...", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled server exception: %s on %s", str(exc), request.url.path, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "detail": "An unexpected error occurred during request processing.",
            },
        )

    # Register API Endpoints
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(api_router)  # Also mount directly at root for convenience

    # Mount static assets for web UI and sample images
    if STATIC_DIR.exists():
        app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="static-web")

    if SAMPLES_DIR.exists():
        app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

    @app.get("/", tags=["Root"])
    async def root_redirect():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "online",
            "docs": "/docs",
            "web_ui": "/app",
            "endpoints": {
                "health": "/health",
                "metadata": "/model/metadata",
                "predict": "/predict (POST)",
                "batch_predict": "/predict/batch (POST)",
            },
        }

    return app


app = create_app()
