from flask import Flask, request, render_template
import joblib

model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

def predict_label(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)
    return pred[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['review_text']
        prediction = predict_label(user_input)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
