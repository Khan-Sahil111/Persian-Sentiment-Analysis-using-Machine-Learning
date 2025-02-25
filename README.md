# 💙 Persian Sentiment Analysis using Machine Learning

## 🚀 Project Overview
This project performs **Sentiment Analysis on Persian Text** using **Machine Learning**. The dataset consists of restaurant reviews from Snappfood, and the goal is to classify them as **Positive** or **Negative**. The model is trained using **TF-IDF vectorization** and multiple machine learning algorithms.

---

## 🔹 Dataset
The dataset used for this project includes Persian text reviews from Snappfood and contains the following columns:

| Column Name      | Description                                  |
|-----------------|----------------------------------------------|
| `comment`       | Original Persian text review                |
| `label`         | Sentiment label (`1 = Positive, 0 = Negative`) |
| `comment_length`| Length of the comment                       |
| `comment_cleaned` | Preprocessed Persian text (stopwords removed, etc.) |

---

## ⚙️ Technologies Used
- **Python** 🐍
- **Scikit-learn** (Machine Learning)
- **NLTK & Hazm** (Persian Text Preprocessing)
- **TF-IDF Vectorization** (Feature Extraction)
- **Joblib** (Model Saving & Loading)

---

## 📀 Model Training & Evaluation
We experimented with multiple classifiers to find the best model:

| Model                     | Accuracy  |
|---------------------------|----------|
| **Logistic Regression**   | 82.52%      |
| **Random Forest**         | 81.54%      |
| **Support Vector Machine (SVM)** | 82.25% |

The best-performing model was **Logistic Regression**, which achieved **82.52% accuracy**.

---

## 🔹 How to Run the Project

### 2⃣ Train the Model
Run the Jupyter Notebook or Python script:
```bash
python train_model.py
```
This will save the trained model as `sentiment_model.pkl` and the TF-IDF vectorizer as `tfidf_vectorizer.pkl`.

### 3⃣ Make Predictions
You can use the trained model to classify new Persian comments:
```python
import joblib

# Load Model & Vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sample Comment
sample_comment = ["این غذا خیلی بد بود"]
sample_vector = vectorizer.transform(sample_comment)

# Predict Sentiment
prediction = model.predict(sample_vector)[0]
print(f"Prediction Value: {prediction}")
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

---

## 📂 Project Structure
```
📝 Persian-Sentiment-Analysis
 ┣  📄 cleaned_snappfood_dataset.csv
 ┣  📄 sentiment_model.pkl
 ┃  📄 tfidf_vectorizer.pkl
 ┣  📄 sentiment_analysis.ipynb
 ┣  📄 requirements.txt
 ┣  📄 README.md
```

---

## 🎯 Future Improvements
✅ Improve accuracy with **Deep Learning** (LSTMs, Transformers)  
✅ Add **More Persian Datasets** to generalize better  
✅ Deploy as a **Web App using Streamlit or Flask**  

---

## 👨‍💻 Author
**Sahil**  
📧 sahilgtk@gmail.com 
👉 www.linkedin.com/in/khan-sahil-228562289    

---

## ⭐ Contribute & Support
If you found this project useful, **feel free to star ⭐ the repository** and share your feedback! PRs and contributions are welcome. 🚀

