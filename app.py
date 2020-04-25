from flask import Flask, render_template, redirect, request
from Predict_that_flair import model, tk, le
import Predict_that_flair


app = Flask(__name__)


@app.route('/')
def basic():
    return render_template("index.html")

@app.route('/',methods = ['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['post_url']
        flair = Predict_that_flair.predict_flair(model,url,tk,le)

    return render_template("index.html", predicted_flair = flair)


if __name__ == '__main__':
    app.run(debug = False,threaded = False)
