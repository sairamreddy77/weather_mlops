from flask import Flask
from routes import weather_blueprint

app = Flask(__name__)
app.register_blueprint(weather_blueprint)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

