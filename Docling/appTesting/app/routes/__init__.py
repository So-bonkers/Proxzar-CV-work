from flask import Blueprint
from .download_routes import download_bp

# You can initialize any shared configurations here
def register_routes(app):
    """
    Registers all route Blueprints to the Flask app.
    """
    app.register_blueprint(download_bp)
