from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Load the trained model and vectorizer
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)
classified_emails = []

@app.route('/')
def home():
    return render_template("index.html", classified_emails=classified_emails, pie_chart=None)

@app.route('/classify', methods=['POST'])
def classify():
    email_text = request.form['email_text']
    vectorized_text = vectorizer.transform([email_text])
    prediction = model.predict(vectorized_text)[0]
    label = "Spam" if prediction == 1 else "Ham"
    classified_emails.append({'Message': email_text, 'Label': label, 'Predicted': prediction})
    
    # Convert to DataFrame
    df_classified = pd.DataFrame(classified_emails)
    spam_count = df_classified[df_classified['Label'] == 'Spam'].shape[0]
    ham_count = df_classified[df_classified['Label'] == 'Ham'].shape[0]
    
    # Pie chart
    plt.figure(figsize=(5, 5))
    plt.pie([spam_count, ham_count], labels=['Spam', 'Ham'], autopct='%1.1f%%', colors=['red', 'green'])
    plt.title("Spam vs Ham Distribution")
    pie_img = io.BytesIO()
    plt.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_url = base64.b64encode(pie_img.getvalue()).decode()
    plt.close()
    
    return render_template("index.html", classification=label, email_text=email_text, 
                           classified_emails=classified_emails, pie_chart=pie_url)

if __name__ == '__main__':
    app.run(debug=True)
