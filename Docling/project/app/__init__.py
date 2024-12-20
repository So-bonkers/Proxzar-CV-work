from flask import Flask
from app.routes import main_blueprint
import os

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder=os.path.abspath("templates"),
        static_folder=os.path.abspath("static")
    )

    # Register Blueprints
    app.register_blueprint(main_blueprint)

    return app
