## 📰 News Classifier using TF-IDF + ML

This is a simple foundation to learn machine learning, based on how news website can understand and categorize articles. This project pulls back the curtain, showing you how to build a news classifier from scratch. In this project we use a classic but powerful technique called TF-IDF (i recommend you to watch this video for a simple explanation, if you want to know the theory behind the vectorization: https://www.youtube.com/watch?v=vZAXpvHhQow) paired with Logistic Regression to teach our model to distinguish between different news topics.

This project demonstrates a **simple but powerful text classification pipeline** using **TF-IDF vectorization** and **machine learning models** (Logistic Regression). Two real-world news datasets are used:

- **AG News Dataset** (4 classes)
- **BBC News Dataset** (5 classes)

In this project we use 2 different datasets to implement two different examples/notebooks for AG News it's a kaggle-style dataset (which means it has been cleaned for open source projects like this!) which we would implement a Multinomial model and for BBC News, a raw dataset that we have to preproccess/clean we use a Logistic Regression approach.

**kaggle links!**

- https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
- https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

---

### What You’ll Learn!

- How to clean and prepare raw text data
- Combine text features like title + content
- Vectorize text with **TF-IDF**
- Train/test split and model training with `scikit-learn`
- Evaluate performance with classification reports

---

### 🗂️ Dataset Overview

| Dataset  | Classes | Format | Source                    |
| -------- | ------- | ------ | ------------------------- |
| AG News  | 4       | CSV    | Open-source, Kaggle-style |
| BBC News | 5       | CSV    | BBC Articles (raw .txt)   |

---

### 📁 Project Structure

```
news-classifier-project/
├── README.md
├── requirements.txt
├── data/
│   ├── ag_news_test.csv
│   └── bbc-news-data.csv
├── notebooks/
│   ├── 01_AG_News_Classifier.ipynb
│   └── 02_BBC_News_Classifier.ipynb
```

---

### How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

---

### (Libraries Used)

- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

---

2. Open either notebook:

```bash
jupyter notebook notebooks/01_AG_News_Classifier.ipynb
```

3. Run all cells from top to bottom. Each notebook is **self-contained** and includes:

   - Data loading
   - Preprocessing
   - TF-IDF vectorization
   - Model training and evaluation

---

### My Model Performance

#### AG News — Logistic Regression

| Class        | Precision | Recall | F1 Score |
| ------------ | --------- | ------ | -------- |
| World        | 0.86      | 0.90   | 0.88     |
| Sports       | 0.94      | 0.96   | 0.95     |
| Business     | 0.84      | 0.84   | 0.84     |
| Sci/Tech     | 0.89      | 0.82   | 0.85     |
| **Accuracy** |           |        | **0.88** |

#### BBC News — Logistic Regression

| Class       | Accuracy  |
| ----------- | --------- |
| All Classes | \~**99%** |

> Final models use `TfidfVectorizer(stop_words='english', max_features=5000)` + `LogisticRegression(max_iter=1000)`

### 📌 Notes

- The AG News file was manually cleaned to match expected column names.
- The BBC dataset required preprocessing to handle messy text content and merge title+content.

Thanks!
