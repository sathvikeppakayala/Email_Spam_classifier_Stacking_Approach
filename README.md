# Email Spam Classifier - Stacking Approach

Welcome to the **Email Spam Classifier** repository! This project utilizes advanced machine learning ensemble techniques—specifically a refined stacking approach with a voting meta-classifier—to classify emails as spam or ham with high accuracy.

## 🚀 Features
- **Stacking & Voting Ensemble**: Combines multiple classifiers for superior prediction.
- **Web Interface**: User-friendly Flask-based interface for real-time classification.
- **Performance Metrics**: Displays accuracy, precision, recall, and F1-score.
- **Visualization**: Generates a pie chart to visualize spam vs ham distribution.

## 🛠️ Technologies Used
- **Python** (3.8+)
- **Flask**: For the web framework.
- **Scikit-learn**: For building machine learning models.
- **Matplotlib & Seaborn**: For data visualization.
- **Pandas & NumPy**: For data handling and preprocessing.
- **NLTK**: For text preprocessing (stopword removal, tokenization).

## 📂 Project Structure
```
Email_Spam_classifier_Stacking_Approach/
├── app.py                # Flask application backend
├── index.html            # HTML template for the frontend
├── spam_classifier.pkl   # Trained stacking classifier model
├── tfidf_vectorizer.pkl  # Trained TF-IDF vectorizer
├── research paper.docx   # Research documentation on the approach
└── README.md             # Project documentation
```

## ⚙️ Installation
1. **Clone the Repository**
```bash
git clone https://github.com/sathvikeppakayala/Email_Spam_classifier_Stacking_Approach.git
cd Email_Spam_classifier_Stacking_Approach
```
2. **Create and Activate Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Ensure `requirements.txt` contains necessary packages like Flask, scikit-learn, pandas, numpy, nltk, matplotlib, seaborn)*

## 📊 Running the Application
```bash
python app.py
```

Navigate to `http://127.0.0.1:5000/` to access the web interface.

## 💡 Usage
1. Input email content into the text area.
2. Click **Classify** to determine if the email is Spam or Ham.
3. View classification results, model performance, and visual insights.

## 📈 Model Overview
- **Base Classifiers**: Logistic Regression, Decision Tree, K-Nearest Neighbors, Multinomial Naïve Bayes, Random Forest, AdaBoost.
- **Meta Classifier**: Voting Classifier (Soft Voting).
- **Stacking Ensemble**: Voting Classifier as the final estimator to optimize performance.
- **Accuracy Achieved**: 98.18%

## 📚 Research Insights
This project is backed by a detailed research study that explores the efficiency and accuracy of ensemble learning models in spam detection. The stacking model with a voting classifier significantly outperforms conventional classifiers.

## 🔧 Future Enhancements
- Integration of transformer-based models like BERT.
- Real-time feedback for model retraining.
- Advanced NLP techniques for better context understanding.
- Reinforcement learning for continuous model improvement.

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository, enhance the code, and submit pull requests.

## 📄 License
This project is licensed under the Creative Commons 4.0  License.

---

> Developed by [Sathvik Eppakayala](https://sathvikeppakayala.github.io)

