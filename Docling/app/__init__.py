from flask import Flask
from app.routes import main_blueprint
from flask_jwt_extended import JWTManager
import os

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__)
    # Set a Secret Key for Sessions
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "super-secret-key")  # Change for production
    
    # Set JWT Secret Key
    app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-jwt-key")

    # Initialize JWT
    jwt = JWTManager(app)

    # Register Blueprints
    from app.routes import main_blueprint
    app.register_blueprint(main_blueprint)

    return app
