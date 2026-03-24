# 📧 Email Spam Classifier

A multi-class email classification system built with **XGBoost** and **SentenceTransformers**, capable of distinguishing between Ham, Spam, Phishing, and Nigerian 419 scam emails with **93.2% overall accuracy**.

---

## 🏆 Model Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| ✅ Ham | 0.91 | 0.99 | **0.95** |
| 🚫 Spam | 1.00 | 1.00 | **1.00** |
| 🎣 Phishing | 0.77 | 0.55 | **0.64** |
| 💰 Nigerian 419 | 0.81 | 0.73 | **0.77** |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** |

> **Best CV F1 Score:** `0.9202 ± 0.0027` across 5 folds

---

## 📁 Project Structure

```
email_spam_classifier/
├── artifacts/                      # Generated after training
│   ├── model.pkl                   # Trained XGBoost model
│   ├── tld_encoders.pkl            # TLD encoders (train/inference consistency)
│   ├── feature_columns.json        # Feature column order (426 features)
│   └── confusion_matrix.png        # Performance visualization
│
├── archive/                        # Training data (not included)
│   ├── CEAS_08.csv                 # Ham/Spam emails
│   ├── Nazario_5.csv               # Phishing emails
│   └── Nigerian_5.csv              # Nigerian 419 scam emails
│
├── preprocessTrainingData.py       # Data loading and TLD encoding
├── emailFeature.py                 # Sender/receiver feature extraction
├── urlFeatureCreation.py           # URL pattern analysis
├── textEmbedding.py                # SentenceTransformer embeddings
├── clearnText.py                   # Text preprocessing utilities
├── train.py                        # Training pipeline with HPO
├── classifier.py                   # XGBoost model and tuning
├── gradioInterface.py              # Web inference interface
└── requirements.txt
```

---

## 🗂️ Data Format

Place your datasets in an `archive/` folder. Each CSV must contain:

| Column | Description |
|---|---|
| `sender` | Sender name and email address |
| `receiver` | Receiver email address |
| `subject` | Email subject line |
| `body` | Full email body text |

- `CEAS_08.csv` — requires a `label` column (`0` = ham, `1` = spam)
- `Nazario_5.csv` — phishing emails (label assigned automatically)
- `Nigerian_5.csv` — Nigerian 419 scam emails (label assigned automatically)

---

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

GPU acceleration is optional but recommended. Check availability with:
```bash
nvidia-smi
```

### 2. Train the Model

```bash
python train.py
```

The training pipeline runs in 5 stages:

```
[1/5] Processing training data...
[2/5] Splitting data into train/test sets...
[3/5] Training model with hyperparameter tuning...   # ~10–30 min
[4/5] Saving model artifacts...
[5/5] Evaluating model on test set...
```

Artifacts are saved to the `artifacts/` directory upon completion.

### 3. Launch Inference Interface

```bash
python gradioInterface.py
```

Opens a Gradio web UI at `http://localhost:7860`. Supports:
- Manual input (sender, receiver, subject, body)
- One-click classification with confidence scores
- Pre-loaded example emails for all four classes

---

## ⚙️ Best Hyperparameters

Found via 50-iteration Bayesian search (`RandomizedSearchCV`):

| Parameter | Value |
|---|---|
| `n_estimators` | 265 |
| `max_depth` | 4 |
| `learning_rate` | 0.1376 |
| `colsample_bytree` | 0.8214 |
| `subsample` | 0.9820 |
| `gamma` | 2.6633 |
| `min_child_weight` | 4 |
| `reg_alpha` | 0.3345 |
| `reg_lambda` | 2.7421 |
| `scale_pos_weight` | 1.7446 |

---

## 🔧 Feature Engineering

The model uses **426 total features** across three categories:

### Email Header Features (`emailFeature.py`)
- Local part length, domain length, special character counts
- Entropy of sender/receiver addresses
- Free provider detection (Gmail, Yahoo, Outlook, etc.)
- Suspicious keyword flags (`admin`, `support`, `verify`, `noreply`)
- TLD encoding (consistent across train and inference via `LabelEncoder`)
- IP address detection in sender field

### URL Features (`urlFeatureCreation.py`)
- Average URL length, special character density
- Digit-to-letter ratio in URLs
- Suspicious keyword detection in URLs
- Redirect and URL shortener detection
- IP address presence in URLs

### Text Embeddings (`textEmbedding.py`)
- 384-dimensional semantic embeddings via `SentenceTransformer`
- Captures meaning of subject + body content

---

## 🐛 Known Issues & Fixes

| # | Bug | Impact | Fix Applied |
|---|---|---|---|
| 1 | `pd.Categorical().codes` for TLD encoding | Different mappings at train vs inference → wrong predictions | `LabelEncoder` with saved state |
| 2 | Boolean features not cast to int | XGBoost can't learn from Python booleans | All booleans → `int(...)` |
| 3 | No model file existence checks | Cryptic errors at inference time | Added upfront validation |
| 4 | Inconsistent label formatting | Poor UX in Gradio interface | Standardized with emojis and warnings |

---

## 🐞 Troubleshooting

**`TLD encoders not found` / `Model not found`**
```bash
python train.py   # Train before running inference
```

**CUDA out of memory**
```python
# In classifier.py:
device='cuda'  →  device='cpu'

# In textEmbedding.py:
batch_size=32  →  batch_size=16
```

**Low Phishing / Nigerian 419 F1 scores**
- These classes have fewer training samples — collect more data
- Increase tuning iterations: set `n_iter=100` in `classifier.py`
- The current model already handles unknown TLDs gracefully (mapped to `-1`)

---

## 📈 Roadmap

- [ ] Add timestamp-based features (hour of day, day of week)
- [ ] Extract phone numbers and currency amounts from body
- [ ] Integrate URL reputation API
- [ ] Fine-tune a BERT/DistilBERT model for text encoding
- [ ] Build an ensemble with XGBoost + transformer classifier
- [ ] Add LIME/SHAP explanations to the Gradio interface

---

## 📄 License

For educational and research use only.
