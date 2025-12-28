from flask import Flask
from routes.yolo_routes import yolo_routes

app = Flask(__name__)
app.register_blueprint(yolo_routes, url_prefix="/yolo")


@app.route("/")
def home():
    return {"message": "YOLO Flask Backend Running"}


if __name__ == "__main__":
    app.run(debug=True)
