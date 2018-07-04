from flask import Flask,redirect,url_for
from model.generate import generate


app = Flask(__name__)


@app.route('/')
def index():
  url = url_for('static', filename='index.html')
  return redirect(url)


if __name__ == '__main__':
  app.run()
