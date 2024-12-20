from flask import Flask
from app.routes import main_blueprint

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__, template_folder="templates")

    # Register Blueprints
    app.register_blueprint(main_blueprint)

    return app
