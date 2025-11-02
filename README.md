```markdown
# 🛡️ AI-Driven Email Threat Detection and Classification

An **AI-based cybersecurity system** that detects and categorizes **phishing, spam, and malicious emails** using **Natural Language Processing (NLP)**, **Transformer embeddings**, and **URL + header feature engineering**.  
Developed as part of the **Cybersecurity Track (Inter-IIIT Nagpur)** challenge.

---

## 📌 Problem Statement

Build an AI-powered email analysis tool that automatically classifies incoming emails into multiple **risk categories**, detects suspicious links, and supports potential real-time alerting.

### **Threat Classes**
| Label | Dataset | Threat Level |
|:------:|:---------|:-------------|
| 1 | Nigerian Fraud | 🚨 Extreme |
| 2 | Nazario | 🔴 Very High |
| 3 | CEAS 08 | 🟠 Medium |
| 4 | SpamAssassin | 🟢 Low |

---

## 🚀 Features

- 🧠 **Multi-Class Classification:** Detects different kinds of malicious/spam emails.  
- 🔍 **URL Extraction:** Identifies and normalizes suspicious links from email bodies.  
- 🧩 **Semantic Understanding:** Uses `SentenceTransformer` for deep sentence embeddings.  
- ⚙️ **Comprehensive Feature Engineering:** Combines textual, structural, and metadata features.  
- 📊 **Visualization:** Displays confusion matrices, accuracy metrics, and model comparisons.  
- 🔒 **Security-Centric:** Designed for email threat detection and cyber defense research.

---

## 🧠 Workflow

![Workflow](WorkFlow.png)

1. **Data Loading:** Import datasets — *Nigerian_Fraud.csv*, *Nazario.csv*, *CEAS_08.csv*, *SpamAssassin.csv*  
2. **Preprocessing:** Clean headers, normalize URLs (`hxxp` → `http`), handle empty bodies.  
3. **Feature Engineering:** Extract URLs, keywords, and sentence embeddings.  
4. **Model Training:** Use **XGBoost**, **Transformer-based encoders**, and classical ML models.  
5. **Evaluation:** Multi-class accuracy, F1-score, and detailed performance reporting.

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3 |
| ML Frameworks | Scikit-learn, XGBoost, PyTorch |
| NLP | Transformers, Sentence-Transformers, NLTK, SpaCy |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |

---

📁 Project Structure
interiiitn_email_spam/
│
├── interiiit.ipynb              # Main Jupyter notebook (end-to-end workflow)
├── archive/                     # Datasets used for training & testing
│   ├── Nigerian_Fraud.csv
│   ├── Nazario.csv
│   ├── CEAS_08.csv
│   └── SpamAssassin.csv
│
├── WorkFlow.png                 # Workflow diagram (optional visual representation)
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation


---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshitgupta04022004/interiiitn_email_spam.git
   cd interiiitn_email_spam
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook interiiit.ipynb
   ```

---

✅ Final Evaluation (from the notebook)
Classification Report
--- Classification Report ---
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4281
           1       0.97      0.99      0.98       666
           2       0.99      0.97      0.98       313
           3       0.99      1.00      1.00      4368
           4       0.96      0.81      0.87       344

    accuracy                           0.99      9972
   macro avg       0.98      0.95      0.96      9972
weighted avg       0.99      0.99      0.99      9972

Individual Metrics (Weighted Avg)

Accuracy: 0.9875

Weighted Precision: 0.9873

Weighted Recall: 0.9875

Weighted F1-Score: 0.9871

Confusion Matrix (Raw)
[[4255    1    1   20    4]
 [   4  657    1    1    3]
 [   5    3  303    0    2]
 [   8    1    0 4355    4]
 [  46   15    1    5  277]]


These numbers are the exact outputs produced and printed by the final evaluation cell in interiiit.ipynb.

📊 Example Predictions
Example Email Snippet	Predicted Label	Interpretation
Win $10,000 by clicking hxxp://claim-now.com	1	Nigerian Fraud (Extreme)
Official newsletter: meeting schedule attached	0	Other / benign
Reset your bank password at https://secure-bank.example	2/3/4 (depends)	Suspicious — check URL/domain

(Exact label choice depends on the pattern that matches; see the notebook for the detailed decision logic.)
---

## 📚 References

* [Sentence Transformers](https://www.sbert.net/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Scikit-learn](https://scikit-learn.org/)
* [XGBoost Docs](https://xgboost.readthedocs.io/)
* [SpamAssassin Public Corpus](https://spamassassin.apache.org/publiccorpus/)
* [Nazario Phishing Dataset](https://monkey.org/~jose/phishing/)

---

## 👨‍💻 Author

**Harshit Gupta**
Engineer | AI & Cybersecurity Enthusiast
📧 [harshitgupta04022004@gmail.com](mailto:harshitgupta04022004@gmail.com)
🌐 [GitHub Profile](https://github.com/harshitgupta04022004)

---

## 💬 Contributions

Contributions, issues, and suggestions are welcome!
If you find this project useful, please ⭐ the repo to support further development.

---
