from flask import Flask
from routes import bp as routes_bp
from dotenv import load_dotenv

import os

load_dotenv() # Load .env file

app = Flask(__name__) # Create Flask app
app.config["SECRET_KEY"] = os.getenv("CONFIG") # for Flask-WTF
app.secret_key = os.getenv("SECRET_KEY") # for session management
app.register_blueprint(routes_bp) # Register routes

if __name__ == "__main__":
    app.run(debug=True)
