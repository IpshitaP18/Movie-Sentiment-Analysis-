# 🎬 Movie Review Sentiment Analysis using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Apps-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-ScikitLearn-orange.svg)

## 📌 Project Overview

Movie Review Sentiment Analysis is a Natural Language Processing (NLP) based Machine Learning project that classifies movie reviews into two categories:

* 😊 Positive Sentiment
* 😞 Negative Sentiment

The system analyzes the text of a movie review, performs preprocessing, converts the text into numerical features using TF-IDF, and uses a trained Machine Learning model to predict the sentiment.

A Streamlit-based web interface is developed to provide an interactive platform where users can enter reviews and view sentiment predictions along with model performance visualizations.

---

# 🚀 Features

* Text preprocessing using NLP techniques
* Removal of:

  * HTML tags
  * URLs
  * Punctuation
  * Numbers
  * Stopwords
* Lemmatization using WordNet
* TF-IDF based feature extraction
* Multiple Machine Learning model comparison
* Sentiment prediction for new reviews
* Interactive Streamlit web application
* Visualization of:

  * Sentiment distribution
  * Confusion matrix

---

# 🧠 Machine Learning Workflow

```
IMDB Movie Reviews Dataset
            |
            ↓
     Text Preprocessing
            |
            ↓
       TF-IDF Vectorization
            |
            ↓
     Train-Test Split
            |
            ↓
 Multiple ML Model Training
            |
            ↓
 Performance Comparison
            |
            ↓
 Best Model Selection
            |
            ↓
 Streamlit Deployment
            |
            ↓
 Positive / Negative Prediction
```

---

# 📂 Project Structure

```
Sentiment Analysis
│
├── app.py                    # Streamlit frontend application
├── train_model.py            # Model training and comparison
├── preprocessing.py          # Dataset preprocessing
├── utils.py                  # Text preprocessing functions
├── requirements.txt          # Required Python libraries
├── README.md                 # Project documentation
├── .gitignore
│
├── dataset/
│   └── IMDB_Dataset.csv      # Dataset (not included in GitHub)
│
├── models/
│   ├── sentiment_model.pkl   # Trained model (generated)
│   └── tfidf.pkl             # TF-IDF vectorizer (generated)
│
└── images/
    └── confusion_matrix.png  # Model evaluation graph
```

---

# 📊 Dataset Description

The project uses the **IMDB Movie Review Dataset** for training and evaluation.

The dataset contains:

* **50,000 movie reviews**
* **2 sentiment classes:**

  * Positive
  * Negative

Dataset columns:

| Column    | Description                         |
| --------- | ----------------------------------- |
| review    | Movie review text                   |
| sentiment | Sentiment label (positive/negative) |

The dataset can be downloaded from:

🔗 **IMDB Dataset of 50K Movie Reviews:**
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

After downloading, place the dataset file as:

```text
dataset/IMDB_Dataset.csv
```

The dataset is then preprocessed by removing unwanted text elements and generating a cleaned dataset for model training.

---

# 🛠️ Technologies Used

## Programming Language

* Python 3.x

## Libraries

| Library       | Purpose                       |
| ------------- | ----------------------------- |
| Pandas        | Data loading and manipulation |
| NumPy         | Numerical operations          |
| NLTK          | Natural Language Processing   |
| BeautifulSoup | Removing HTML tags            |
| Scikit-learn  | Machine Learning algorithms   |
| Matplotlib    | Data visualization            |
| Streamlit     | Web application development   |

---

# 🤖 Machine Learning Models Compared

The following algorithms were evaluated:

| Algorithm               |
| ----------------------- |
| Logistic Regression     |
| Multinomial Naive Bayes |
| Support Vector Machine  |
| Random Forest           |

Models were compared using:

* Accuracy
* Precision
* Recall
* F1-score

The best performing model was selected and saved for final prediction.

---

# 🔎 Text Preprocessing Steps

The review text goes through the following steps:

1. Convert text to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove punctuation
5. Remove numbers
6. Remove stopwords
7. Apply lemmatization

Example:

Before:

```
The movie was AMAZING!!! I watched it twice.
```

After:

```
movie amazing watched twice
```

---

# 📈 Feature Extraction

## TF-IDF Vectorization

Machine learning algorithms cannot directly understand text.

TF-IDF converts textual reviews into numerical vectors based on word importance.

Example:

```
"excellent movie"

↓

[0.45, 0.78, 0.12 ...]
```

These numerical features are used for model training.

---

# 🖥️ Running the Project Locally

## 1. Clone Repository

```bash
git clone <repository-url>
```

Move into project directory:

```bash
cd "Sentiment Analysis"
```

---

## 2. Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

---

## 3. Add Dataset

Download the IMDB Movie Review Dataset from:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the dataset file inside:

```
dataset/IMDB_Dataset.csv
```

---

## 4. Preprocess Dataset and Train the Model

First, run the preprocessing script:

```bash
python preprocessing.py
```

This step performs:

* Loading the IMDB dataset
* Removing missing values and duplicate records
* Cleaning movie reviews using NLP techniques
* Removing HTML tags, URLs, punctuation, and stopwords
* Applying lemmatization
* Creating the cleaned dataset

Generated file:

```
dataset/cleaned_IMDB_Dataset.csv
```

Note:
`utils.py` contains reusable text preprocessing functions and is automatically imported by other files. It does not need to be executed separately.

---

Run the model training script:

```bash
python train_model.py
```

This will:

- Convert text into numerical features using TF-IDF
- Train multiple Machine Learning algorithms
- Compare model performance using evaluation metrics
- Select the best-performing model
- Save the trained model
- Generate the confusion matrix
- Generate the sentiment distribution graph

Generated files:

```
models/
    sentiment_model.pkl
    tfidf.pkl
```

and

```
images/
    confusion_matrix.png
    sentiment_distribution.png
```

---

## 5. Run Streamlit Application

Start the frontend:

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 🎥 Application Workflow

1. User enters a movie review through the Streamlit interface.
2. Review text is cleaned using NLP preprocessing techniques.
3. Cleaned text is converted into TF-IDF numerical features.
4. The selected Machine Learning model predicts the sentiment.
5. The result is displayed as Positive or Negative.
6. The application also displays sentiment distribution and model evaluation visualization.

---

# 📌 Sample Predictions

### Positive Review

Input:

```
This movie was amazing. The acting and storyline were excellent.
```

Output:

```
Positive 😊
```

### Negative Review

Input:

```
The movie was boring and the acting was terrible.
```

Output:

```
Negative 😞
```

---

# 📊 Evaluation Metrics

The model performance is evaluated using:

### Accuracy

Measures the percentage of correctly classified reviews.

### Precision

Measures how many predicted positive reviews are actually positive.

### Recall

Measures how many actual positive reviews are correctly identified.

### F1 Score

Combination of precision and recall.

---

# 🔮 Future Scope

* Implement deep learning models such as LSTM and BERT
* Extend classification to multiple emotions
* Include user ratings and contextual information
* Integrate with movie recommendation systems
* Deploy the application on cloud platforms

---

# 📜 License

This project is developed for educational purposes.
