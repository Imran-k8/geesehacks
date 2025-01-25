from app import app
from flask import request, jsonify


@app.routes("/home")
def home():
    return