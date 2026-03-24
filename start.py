"""
PayChat — One-Command Demo Launcher (Windows)
Runs: generate data → train model → start FastAPI → (optional) ngrok tunnel

Usage:
  python start.py              # train + serve locally
  python start.py --ngrok      # train + serve + public URL via ngrok
  python start.py --serve-only # skip training, just start server
"""

import subprocess
import sys
import time
import os
import json
import argparse
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print(f"\n▶ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str))
    return result


def install_deps():
    print("\n" + "="*60)
    print("  [1/4] Installing dependencies")
    print("="*60)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])


def generate_data():
    print("\n" + "="*60)
    print("  [2/4] Generating training data")
    print("="*60)
    run([sys.executable, "generate_data.py"], cwd="data")


def train_model():
    print("\n" + "="*60)
    print("  [3/4] Training your model (~10 min on CPU, ~2 min on GPU)")
    print("  Tip: the accuracy report will show at the end")
    print("="*60)
    run([sys.executable, "train.py"], cwd="model")


def start_server():
    print("\n" + "="*60)
    print("  [4/4] Starting inference API at http://localhost:8000")
    print("="*60)
    # Pass absolute MODEL_DIR so the API finds the model regardless of cwd
    model_dir = str(Path("model/saved_model").resolve())
    env = {**os.environ, "MODEL_DIR": model_dir}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="api",
        env=env,
    )
    time.sleep(3)
    return server


def start_ngrok():
    print("\n" + "="*60)
    print("  [+] Starting ngrok tunnel")
    print("  Get your token at: https://ngrok.com (free)")
    print("="*60)
    try:
        ngrok = subprocess.Popen(
            ["ngrok", "http", "8000", "--log=stdout"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        time.sleep(3)

        # Get public URL from ngrok API
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:4040/api/tunnels").read()
        tunnels = json.loads(resp)
        url = tunnels["tunnels"][0]["public_url"]
        print(f"\n  ✓ Public URL: {url}")
        print(f"  ✓ Share this with your friend!")

        # Patch demo HTML
        html_path = Path("demo/paychat_demo.html")
        if html_path.exists():
            html = html_path.read_text()
            html = html.replace('const API_URL = "YOUR_API_URL_HERE"', f'const API_URL = "{url}"')
            html_path.write_text(html)
            print(f"  ✓ Demo HTML updated with your ngrok URL")
            print(f"\n  → Open demo/paychat_demo.html in a browser (or drop to Netlify)")

        return ngrok, url
    except FileNotFoundError:
        print("  ✗ ngrok not found. Install: winget install ngrok")
        print("  Running without ngrok (localhost only)")
        return None, "http://localhost:8000"
    except Exception as e:
        print(f"  ✗ ngrok error: {e}")
        return None, "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrok", action="store_true", help="Create ngrok tunnel for sharing")
    parser.add_argument("--serve-only", action="store_true", help="Skip training, just serve")
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  PayChat Money Detector — Launch Script")
    print("█"*60)

    install_deps()

    if not args.serve_only:
        # Check if data exists
        if not Path("data/train.json").exists():
            generate_data()
        else:
            print("\n  Skipping data generation (train.json already exists)")

        # Check if model exists
        if not Path("model/saved_model/config.json").exists():
            train_model()
        else:
            print("\n  Skipping training (model already trained)")
            print("  To retrain: delete model/saved_model/ and re-run")

    server = start_server()

    if args.ngrok:
        ngrok, url = start_ngrok()
    else:
        url = "http://localhost:8000"
        # Auto-patch demo HTML with localhost URL
        html_path = Path("demo/paychat_demo.html")
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            if "YOUR_API_URL_HERE" in html or '"http' in html:
                patched = html.replace('const API_URL = "YOUR_API_URL_HERE"', f'const API_URL = "{url}"')
                # Also replace any previously set URL
                import re as _re
                patched = _re.sub(r'const API_URL = "https?://[^"]*"', f'const API_URL = "{url}"', patched)
                html_path.write_text(patched, encoding="utf-8")
                print(f"\n  ✓ Demo HTML patched with {url}")
        print(f"\n  API running at: {url}")
        print(f"  → Open demo/paychat_demo.html in your browser")

    print("\n" + "="*60)
    print(f"  ✓ API: {url}/health")
    print(f"  ✓ Docs: {url}/docs")
    print(f"  ✓ Demo: open demo/paychat_demo.html")
    print("="*60)
    print("\n  Press Ctrl+C to stop\n")

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.terminate()


if __name__ == "__main__":
    main()
