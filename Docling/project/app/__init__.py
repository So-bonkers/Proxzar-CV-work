from flask import Flask
from app.routes import main_blueprint

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__, template_folder=r"C:\Users\kshubhan\Documents\GitHub\Proxzar-CV-work\Docling\project\templates", static_folder=r"C:\Users\kshubhan\Documents\GitHub\Proxzar-CV-work\Docling\project\static")

    # Register Blueprints
    app.register_blueprint(main_blueprint)

    return app
