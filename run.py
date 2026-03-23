"""Project entry point.

Use this file when running the app locally with:
    python run.py
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in {"1", "true", "yes"}
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=debug)
