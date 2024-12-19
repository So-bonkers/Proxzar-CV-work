from flask import Blueprint
from .download_routes import download_blueprint
from .ingestion_routes import ingestion_blueprint
from .extraction_routes import extraction_blueprint

# You can initialize any shared configurations here
def register_routes(app):
    """
    Registers all route Blueprints to the Flask app.
    """
    app.register_blueprint(download_blueprint)
    app.register_blueprint(ingestion_blueprint)
    app.register_blueprint(extraction_blueprint)
