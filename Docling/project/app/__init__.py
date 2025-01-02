# __init__.py
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

    # Centralized error handling
    @app.errorhandler(404)
    def not_found_error(e):
        return {"error": "Resource not found"}, 404

    @app.errorhandler(500)
    def internal_error(e):
        return {"error": "Internal server error"}, 500

    return app