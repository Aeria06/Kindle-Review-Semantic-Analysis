# Kindle-Review-Semantic-Analysis


Project Overview
This project focuses on performing sentiment analysis on Kindle product reviews. The goal is to classify reviews as either positive (overall rating >= 3) or negative (overall rating < 3) using various text vectorization techniques and a Naive Bayes classifier.

Dataset
The dataset used is kindle_reviews.csv, containing various attributes of Kindle reviews, including reviewText and overall (rating).

Key Steps
Data Loading and Initial Inspection: Loaded the kindle_reviews.csv into a pandas DataFrame and inspected its initial structure and columns.
Feature Selection: Selected reviewText and overall columns for the sentiment analysis.
Target Variable Transformation: Converted the overall rating (1-5) into a binary sentiment label:
1 for positive reviews (overall rating >= 3)
0 for negative reviews (overall rating < 3)
Text Preprocessing and Cleaning:
Converted all review text to lowercase.
Removed HTML tags, URLs, and non-alphabetic characters.
Removed English stopwords.
Applied lemmatization to reduce words to their base form.
Data Sampling: To manage computational resources, a small sample of 10,000 reviews (data_small) was used for training and testing.
Train-Test Split: The dataset was split into training (80%) and testing (20%) sets.
Text Vectorization:
Bag-of-Words (BoW): CountVectorizer was used to transform text data into numerical feature vectors, representing word frequencies.
TF-IDF (Term Frequency-Inverse Document Frequency): TfidfTransformer was applied on top of the CountVectorizer output to weigh word importance, providing more nuanced feature representations.
Model Training: Gaussian Naive Bayes classifiers were trained on both BoW and TF-IDF vectorized data.
Model Evaluation: The performance of both models was evaluated using accuracy score and confusion matrices.
Results
After correcting the TF-IDF vectorization, the models were re-evaluated. The accuracies are as follows:

Bag-of-Words (BoW) Model Accuracy: 0.757
Confusion Matrix: [[ 62, 131], [355, 1452]]
TF-IDF Model Accuracy: 0.7575
Confusion Matrix: [[ 63, 130], [355, 1452]]
Both models achieved very similar accuracies, indicating that for this specific dataset and Naive Bayes classifier, the additional weighting of TF-IDF did not significantly improve performance over simple word counts. The slight difference (0.0005) suggests minimal impact.

Dependencies
pandas
scikit-learn
nltk
beautifulsoup4
re
To install the necessary libraries, you can use pip:

pip install pandas scikit-learn nltk beautifulsoup4
For nltk resources (stopwords, wordnet), you'll also need to download them within Python:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

