from app import app


@app.routes("/home")
def home():
    return