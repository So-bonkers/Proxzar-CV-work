from flask import Flask
from config.config_loader import load_config
from routes.ingestion_routes import ingestion_blueprint
from routes.extraction_routes import extraction_blueprint
from routes.download_routes import download_blueprint

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config = load_config("config.json")

# Register blueprints
app.register_blueprint(ingestion_blueprint)  # Register ingestion routes
app.register_blueprint(extraction_blueprint)  # Register extraction routes
app.register_blueprint(download_blueprint)  # Register download routes

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
