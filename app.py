# app.py

from flask import Flask, request, render_template
from fake_news import FakeNewsDetector

app = Flask(__name__)

# Initialize the Fake News Detector
detector = FakeNewsDetector(r'E:\fake_new_prediction\news_dataset.csv')


@app.route('/prediction',methods=['GET', 'POST'])
def news_prediction():
    accuracy = None
    prediction = None
    selected_model = 'logistic'  # Default model

    if request.method == 'POST':
        selected_model = request.form.get('model')
        text_to_predict = request.form.get('news_text')

        # Only perform prediction and evaluation, no dataset loading or reprocessing
        if text_to_predict:
            prediction = detector.predict(text_to_predict, selected_model)

        accuracy = detector.evaluate(selected_model)

    return render_template('result.html', accuracy=accuracy, prediction=prediction, selected_model=selected_model)

@app.route('/news_prediction', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

