from flask import Flask

SECRET_MESSAGE = "You found the secret message!"
app = Flask(__name__)

@app.route('/')
def get_secret_message():
    return SECRET_MESSAGE