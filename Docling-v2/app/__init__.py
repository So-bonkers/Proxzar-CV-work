from flask import Flask
from flask_jwt_extended import JWTManager
import os

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Configuration
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "super-secret-key")  
    app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-jwt-key")

    # Initialize JWT
    jwt = JWTManager(app)

    # Register Blueprints
    from app.routes import main_blueprint
    app.register_blueprint(main_blueprint)

    return app