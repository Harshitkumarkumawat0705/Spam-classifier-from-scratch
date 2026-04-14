# SPAM EMAIL CLASSIFIER (FROM SCRATCH)

## Overview
This project implements a Spam Email Classifier from scratch using the Naive Bayes algorithm, without using machine learning libraries such as scikit-learn.

The model classifies messages into:
- Spam
- Ham (Not Spam)

---

## Features
- Built completely from scratch
- No ML libraries used for modeling
- Custom implementation of:
  - Text preprocessing
  - Tokenization
  - Word frequency counting
  - Laplace smoothing
  - Log probability computation

---

## Dataset
The model is trained on the SMS Spam Collection Dataset.

Labels:
- spam → unwanted or promotional messages  
- ham → normal messages  

---

## Methodology

### 1. Data Preprocessing
- Convert text to lowercase  
- Remove punctuation  
- Tokenize into words  

### 2. Train-Test Split
- Data is shuffled  
- 80% training data  
- 20% testing data  

### 3. Probability Calculation
- Compute prior probabilities:
  - P(spam)
  - P(ham)

- Count word frequencies separately for:
  - Spam messages
  - Ham messages

---

### 4. Naive Bayes Classification

Formula:

P(Class | Message) ∝ P(Class) × Π P(Word | Class)

Using log probabilities:

log(P(Class | Message)) = log(P(Class)) + Σ log(P(Word | Class))

---

### 5. Laplace Smoothing

P(word | class) = (count + α) / (total_words + α × vocabulary_size)

Where:
- α = 1

---

## Evaluation
- Model is tested on unseen data  
- Accuracy is calculated  
- Final Accuracy: 96.86%
- Misclassified messages are displayed  

---

## Tech Stack
- Python  
- Pandas  
- NumPy  
