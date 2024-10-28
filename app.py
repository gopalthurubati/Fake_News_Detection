# app.py

from flask import Flask, request, render_template
from fake_news import FakeNewsDetector

app = Flask(__name__)

# Initialize the Fake News Detector
model = FakeNewsDetector(r'E:\fake_new_prediction\news_dataset.csv', model_type = 'svm')

@app.route('/news_prediction')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def make_prediction():
    input_data = request.form['input_data']
    prediction = model.predict(input_data)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
