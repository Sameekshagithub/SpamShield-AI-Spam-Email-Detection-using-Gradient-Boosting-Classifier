# 🛡️ SpamShield AI — Intelligent Spam Email Detection System

> **Gradient Boosting Classifier · TF-IDF Vectorizer · Flask Web App**

---

## 📁 Project Structure

```
SpamShield-AI/
├── SpamShield_AI_Spam_Email_Detection.ipynb   ← Jupyter Notebook (complete ML project)
├── app.py                                      ← Flask Web Application (single file)
├── spamshield_model.pkl                        ← Saved model (generated after running notebook)
├── tfidf_vectorizer.pkl                        ← Saved vectorizer (generated after running notebook)
├── spam_ham_distribution.png                   ← EDA chart (generated)
├── message_length_analysis.png                 ← EDA chart (generated)
├── model_accuracy_comparison.png               ← Model comparison chart (generated)
├── confusion_matrix.png                        ← Evaluation chart (generated)
└── README.md                                   ← This file
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk flask jupyter
```

### Step 2: Run the Jupyter Notebook

Open and run **all cells** in:
```
SpamShield_AI_Spam_Email_Detection.ipynb
```

This will:
- Download and preprocess the dataset
- Train the Gradient Boosting model
- Generate visualization charts
- Save `spamshield_model.pkl` and `tfidf_vectorizer.pkl`

### Step 3: Launch the Flask Web App

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## 🧠 Algorithm

| Property       | Value                             |
|----------------|-----------------------------------|
| Algorithm      | Gradient Boosting Classifier      |
| Features       | TF-IDF (5000 features, bigrams)   |
| Text Cleaning  | Lowercase, remove noise, stemming |
| Test Accuracy  | ~97–98%                           |
| Dataset        | SMS Spam Collection (~5,574 rows) |

---

## 🌐 Web App Features

- 📧 Email text input area
- 🔍 Real-time spam/ham prediction
- 📊 Model confidence score with progress bar
- 📋 4 sample emails to try (2 spam + 2 ham)
- ⚙️  Preprocessing pipeline visualization
- 💡 Color-coded result cards (red = spam, green = ham)
- ⌨️  Keyboard shortcut: `Ctrl+Enter` to analyze

---

## 📊 Notebook Structure

1. Project Introduction
2. Problem Statement
3. Objective
4. Dataset Information
5. Import Libraries
6. Data Loading
7. Exploratory Data Analysis
8. Data Preprocessing & Text Cleaning
9. Feature Engineering (TF-IDF)
10. Model Building (Gradient Boosting)
11. Model Evaluation
12. Prediction System
13. Save Model with Pickle
14. Flask App Code
15. Final Conclusion

---

## 🛠️ Libraries Used

- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- `nltk` — NLP (stopwords, stemming, tokenization)
- `scikit-learn` — ML models, TF-IDF, evaluation
- `flask` — web application
- `pickle` — model serialization
