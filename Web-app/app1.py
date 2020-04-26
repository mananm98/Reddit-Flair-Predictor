from flask import Flask, request, jsonify, render_template, make_response, redirect
import Predict_that_flair

app1 = Flask(__name__)

@app1.route('/')
def basic():
    return render_template("index.html")

@app1.route('/',methods = ['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['post_url']
        flair = Predict_that_flair.predict_flair(url)

    return render_template("index.html", predicted_flair = flair)

@app1.route('/automated_testing',methods = ['POST'])
def yu():
    if request.method == 'POST':
        data = request.get_data(as_text = True)
        data = data.split()[5:-1]
        predictions = {}
        for i in range(len(data)):
            predictions[data[i]] = Predict_that_flair.predict_flair(data[i])

        predictions = jsonify(predictions)
        return make_response(predictions)


if __name__ == '__main__':
      app1.run(debug = False,threaded = False)
