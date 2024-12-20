from flask import Flask
from app.routes import main_blueprint

CONFIG_FILE = "config.json"

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__, template_folder="Docling/project/templates", static_folder="Docling/project/static")

    # Register Blueprints
    app.register_blueprint(main_blueprint)

    return app
