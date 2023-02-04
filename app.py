
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import load
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import module



# Create flask app
app = Flask(__name__)

vector = load("vectors.joblib")
model = load("model.joblib")
stop_words = stopwords.words('english')
wordnet_lem = WordNetLemmatizer()

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    
    text=[request.form.get("text")]
    
    df = pd.DataFrame(data=text, columns=['text'])
    df['text'] = df['text'].apply(module.cleaning)
    # remove stop word: 
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df['text'] = df['text'].apply(wordnet_lem.lemmatize)
    #prediction
    vec = vector.transform(df['text'])
    prediction = model.predict_proba(vec)
    prediction = int(prediction[0])
    if prediction >0:
        prediction="positive"
    else:
        prediction = "negative"
    return render_template("index.html", prediction_text = "sentiment: {} ".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
