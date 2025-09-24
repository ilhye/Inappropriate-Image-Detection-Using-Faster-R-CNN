from flask import Flask
from routes import bp as routes_bp
import sys
import os


app = Flask(__name__)
app.config["SECRET_KEY"] = "SECRETKEY"
app.secret_key = "random_string"  
app.register_blueprint(routes_bp)

if __name__ == "__main__":
    app.run(debug=True)
