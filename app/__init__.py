"""Application factory for the ACR MRI QA project."""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

# Global extension objects. They are initialized later inside create_app().
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


def _should_initialize_database(app: Flask) -> bool:
    """Return True when the app should run ``db.create_all()`` on startup.

    This is safe for local development and tests. In production containers we default
    to initializing only once to avoid concurrent startup races when multiple workers
    import the app at the same time.
    """
    flag = str(os.getenv("AUTO_CREATE_DB", "1")).lower() in {"1", "true", "yes"}
    return bool(flag or app.config.get("TESTING"))



def create_app(test_config: dict | None = None) -> Flask:
    """Create and configure the Flask application."""
    load_dotenv()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Core security and storage settings.
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Upload settings.
    upload_folder = os.getenv("UPLOAD_FOLDER", "instance/uploads")
    app.config["UPLOAD_FOLDER"] = str((Path(app.root_path) / ".." / upload_folder).resolve())
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("UPLOAD_MAX_MB", "1024")) * 1024 * 1024

    # Optional AWS settings. The app still works when these are blank.
    app.config["AWS_REGION"] = os.getenv("AWS_REGION", "")
    app.config["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
    app.config["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    app.config["AWS_S3_BUCKET"] = os.getenv("AWS_S3_BUCKET", "")
    app.config["AWS_S3_PREFIX"] = os.getenv("AWS_S3_PREFIX", "mri-qa")
    app.config["AWS_S3_ENABLED"] = str(os.getenv("AWS_S3_ENABLED", "0")).lower() in {"1", "true", "yes"}

    if test_config:
        app.config.update(test_config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    # Import models after extensions are configured.
    from app.models import User  # noqa: F401

    with app.app_context():
        if _should_initialize_database(app):
            db.create_all()

    # Register routes.
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    @app.errorhandler(413)
    def upload_too_large(_error):
        """Show a friendly error when the upload exceeds the configured size."""
        return (
            render_template(
                "error.html",
                title="Upload too large",
                message=(
                    "The uploaded data is larger than the server limit. "
                    "Increase UPLOAD_MAX_MB in your .env file or upload a smaller dataset."
                ),
            ),
            413,
        )

    return app
