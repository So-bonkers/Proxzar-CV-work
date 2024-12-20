from flask import Flask
from app.routes import main_blueprint
from app.logging_setup import setup_logging

def create_app():
    app = Flask(__name__)
    setup_logging()

    # Register Blueprints
    app.register_blueprint(main_blueprint)

    return app
