"""NutriFresh AI Command Line Interface (CLI)."""

import argparse
import json
import sys
from pathlib import Path
import uvicorn

from src.config import settings
from src.model.inference import SpoilDetectionEngine
from src.utils.logger import logger


def run_predict(args):
    """Predict freshness of a single image file."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing freshness for: {image_path.name}...")
    engine = SpoilDetectionEngine()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = engine.predict_image(image_bytes)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 45)
        print(f" Prediction:  {result['prediction'].upper()}")
        print(f" Fresh:       {'Yes' if result['is_fresh'] else 'No'}")
        print(f" Confidence:  {result['confidence_percentage']}%")
        print(f" Latency:     {result['latency_ms']} ms")
        print(" Probabilities:")
        for label, prob in result["probabilities"].items():
            print(f"   - {label:<10}: {prob * 100:.2f}%")
        print("=" * 45 + "\n")


def run_evaluate(args):
    """Evaluate an entire directory of images and export results."""
    dir_path = Path(args.dir)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory not found at '{dir_path}'", file=sys.stderr)
        sys.exit(1)

    engine = SpoilDetectionEngine()
    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [p for p in dir_path.glob("*.*") if p.suffix.lower() in supported_exts]

    if not image_files:
        print(f"No valid image files found in '{dir_path}'.")
        return

    print(f"Found {len(image_files)} images in '{dir_path}'. Running batch evaluation...\n")
    results = []

    for img_p in image_files:
        try:
            with open(img_p, "rb") as f:
                img_bytes = f.read()
            res = engine.predict_image(img_bytes)
            res["filename"] = img_p.name
            results.append({"filename": img_p.name, "success": True, "data": res})
            print(f"[{res['prediction']:<6}] {img_p.name:<30} ({res['confidence_percentage']}%)")
        except Exception as exc:
            results.append({"filename": img_p.name, "success": False, "error": str(exc)})
            print(f"[ERROR ] {img_p.name:<30} - {exc}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults successfully exported to '{out_path}'.")


def run_server(args):
    """Launch the FastAPI development/production web server."""
    host = args.host or settings.HOST
    port = args.port or settings.PORT
    reload = args.reload

    print(f"Starting NutriFresh AI API server on http://{host}:{port}...")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="nutrifresh",
        description="NutriFresh AI - Food Freshness & Spoilage Classification CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict freshness of a single image")
    predict_parser.add_argument("--image", "-i", required=True, help="Path to input image file")
    predict_parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Batch evaluate all images in a folder")
    eval_parser.add_argument("--dir", "-d", required=True, help="Path to directory containing images")
    eval_parser.add_argument("--output", "-o", help="Optional path to save JSON results")

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI web server")
    serve_parser.add_argument("--host", default=settings.HOST, help="Host binding IP")
    serve_parser.add_argument("--port", "-p", type=int, default=settings.PORT, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "serve":
        run_server(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
