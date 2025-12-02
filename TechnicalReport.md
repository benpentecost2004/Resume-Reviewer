# Resume Reviewer

# Overview
The dataset used to train our model for this project is a Resume Screening Dataset from huggingface.co. The goal is to build a machine learning pipeline capable of evaluating resumes and predicting recruiter decisions for a given role based on a candidates resume.

# Contributors 
**Alex Merlo**  
amerlo@oakland.edu  
**Ben Pentecost**  
bpentecost@oakland.edu

# Table of Contents
1. [Overview](#overview)  
2. [Contributors](#contributors)  
3. [Data Availability](#data-availability)  
   - [Input Data](#input-data)  
   - [Model Data](#model-data)  
4. [Design Specifications](#design-specifications)  
5. [Approach](#approach)  
   - [Naive Bayes](#naive-bayes)  
     - [Why Naive Bayes](#why-naive-bayes)  
     - [Model Design](#model-design)  
     - [Prediction](#prediction)  
   - [Logistic Regression](#logistic-regression)  
     - [Why Logistic Regression](#why-logistic-regression)  
     - [Model Design](#model-design-1)  
       - [Feature Selection](#feature-selection)  
         - [L1 Regularization (TF-IDF)](#l1-regularization-tf-idf)  
       - [Regression Formula](#regression-formula)  
       - [Logit Function](#logit-function)  
       - [Prediction](#prediction-1)  
       - [Model Interpretation](#model-interpretation)  
6. [Validation Design](#validation-design)  
7. [Technical Design](#technical-design)  
   - [Machine Learning Implementation](#machine-learning-implementation)  
   - [Feature Engineering](#feature-engineering)  
   - [Tools](#tools)  
   - [Design Considerations](#design-considerations)  
8. [Experimental Setup](#experimental-setup)  
   - [Evaluation Metrics](#evaluation-metrics)  
9. [References](#references)



# Data Availability
**Input Data**
- Collected through an interface where users are able to upload their resumes
- PDF data collected using python frameworks
- All text from resumes are stored in pandas tables 
- Will be cleaned and transformed using python and pandas functions 
**Model Data**
- Downloaded directly from Huggingface using the dataset library or manual export
Format: CSV
# Design Specifications
![Design Specs](./Design%20Specs.png)

# Approach
Since we are using a labeled dataset we will be using a supervised learning approach.
We will test the Naive Bayes model and logistic regression in terms of accuracy to predict the chance someone will be hired based on their resume  
## Naive Bayes
### Why Naive Bayes
The Naive Bayes model will provide a baseline performance for predicting whether a candidate will be hired. It will provide a comprehensive view on how individual features, such as skills and experience, contribute to the hiring decision. 
- **Tokenize**  
    - Make words lowercase
    - Remove special characters
- **Convert to features (BoW)**
    - Binary representation of whether a word appears or not

### Model Design
$\boldsymbol{P(C_k \mid x, T) = \dfrac{P(C_k \mid T)\,P(x \mid C_k, T)}{P(x \mid T)}}$
- $C_k$ = Class (Hire / Reject)  
- $x$ = Features extracted from the candidate's resume (skills, experience, education, etc.)  
- $T$ = Job title / position  

$P(C_k, T)$ = Probability the candidate is hired for job position T

$P(x | C_k, T)$ = Probability the given feature $x$ correlates to a candidate being hired for job position T

$P(x | T)$ = Probability of the combination of features for job title regardless of hired or not

### Prediction
Naive Bayes uses the max posterior probability calculated by the formula 
#### $\boldsymbol{\hat{C} = \arg\max_{C_k} P(C_k \mid x, T)}$
This provides the likelihood a candidate is hired to a specific position

## Logistic Regression
### Why Logistic Regression
Logistic regression is able to predict the probability a candidate is hired for a specific job title but unlike Naive Bayes it does not assume independence between features. This makes it more suited to our dataset which will have correlated columns such as skills and job title while also having a binary outcome.
## Model Design
### Feature Selection
#### L1 Regularization (TF-IDF) 
**Goal:** Identify the weights of all keywords in each section and apply L1 regularization for feature selection  
**Reason for implementation:** Offers improved accuracy and identify factors that contribute to the hiring decision. As stated in Improving Native Language Identification with TF-IDF Weighting (Gebre et al.)
**Implementation:** Uses scikit-learn prebuilt function to preform logistic regression on the normalized dataset with l1 penalty applied.
**Output:** Minimized prediction error - Non zero features represent important features that need to be considered
### Regression Formula
The logistic regression model estimates the probability that a candidate is hired based on selected features:
#### Logit Function
The logit function transforms a probability into **log-odds**, allowing a linear relationship with the input features:
### $logit(P) = log\dfrac{p(x)}{1 - p(x)}$  
The probability can be obtained by applying the inverse of the logit function (the sigmoid function):
### $p(x) = \frac{1}{1 + e^{-z}}$
$z = (\beta_0 + \beta_1 x +\beta_2 x + ... +\beta_n x)$  
Where:  
- $p(x)$ = probability that the candidate is hired  
- $x_1, x_2, \dots, x_n$ = input features (skills, experience, education, certifications, job role, etc.)  
- $\beta_0$ = intercept term  
- $\beta_1, \dots, \beta_n$ = coefficients learned by the model  
**Intuition:**  
- Positive logit → higher probability of being hired  
- Negative logit → lower probability of being hired  
- Logit = 0 → probability = 0.5 (threshold for classification)
### Prediction
The model predicts a candidate's hiring outcome based on the probability $p(x)$:
- Classify as **Hire** if $p(x) > 0.5$  
- Classify as **Reject** if $p(x) \leq 0.5$  
### Model Interpretation
- Coefficients ($\beta_i$) indicate the influence of each feature on the hiring probability.  
  - Positive coefficient → feature increases the likelihood of being hired  
  - Negative coefficient → feature decreases the likelihood of being hired  
- Features selected by Boruta ensure that only relevant predictors are included, improving model accuracy and interpretability.

## Validation Design
To ensure the performance of both Naive Bayes and Logistic Regression models, the dataset will be split and evaluated using standard machine learning validation techniques.

### Data Splitting
- **Training set:** 70% of the data for model training  
- **Test set:** 30% of the data for evaluation  
- Use **k-fold cross-validation** to reduce variance and improve reliability of results  
  
### Validation Procedure
1. Preprocess features (encode categorical variables).  
2. Train **Naive Bayes** on the training set.  
3. Train **Logistic Regression** on the same training set (using features selected by L1 Regularization).  
4. Evaluate both models on the test set using the metrics above.  
5. Compare results to determine which model is better for predicting recruiter decisions.  
6. Analyze feature importance (from Boruta and regression coefficients) to understand which candidate attributes influence hiring outcomes.


## Technical Design

### Machine Learning Implementation
- **Naive Bayes:** Implemented from scratch using Python.  
  - Handles tokenization, Bag-of-Words feature extraction, probability calculation, and classification.  
  - Uses Laplace smoothing to handle unseen words in resumes.  

- **Logistic Regression:** Implemented from scratch using Python.  
  - Computes coefficients via gradient descent optimization.  
  - Uses the sigmoid function to convert linear combinations of features into probabilities.  
  - Prediction threshold set at 0.5 for Hire/Reject classification.  

### Feature Engineering
- **Text preprocessing:** Tokenization, lowercase conversion, and removal of special characters for skills, education, and certifications.  
- **Feature encoding:** Convert categorical variables (Job Role, Education, Certifications, Skills) into numerical values.  
- **Feature selection:** Boruta algorithm applied to identify relevant predictors before logistic regression.  

### Tools
- **Python 3.x:** Programming language for all implementations.  
- **Jupyter Notebook:** Interactive development, testing, and visualization of models.  
- **Git / GitHub:** Version control and collaboration.  
- **Matplotlib / Seaborn:** Visualize feature distributions, model outputs, and performance metrics.  

### Design Considerations
- Modular code structure: separate scripts for preprocessing, model implementation, training, and evaluation.  




# Experimental Setup
The research question we aim to address is figuring out what compenets of a resume are important for recruiter decisions. 

### Evaluation Metrics
- **Accuracy:** Percentage of correct predictions (Hire / Reject)  
- **Precision:** How many candidates predicted as Hire were actually hired  
- **Recall:** How many actual hires were correctly predicted  
- **F1-score:** Harmonic mean of precision and recall, balancing false positives and false negatives  
- **ROC-AUC:** Measures overall model performance across all classification thresholds

# References
Kursa, Miron B., and Witold R. Rudnicki. “Feature Selection with the Boruta Package.” Journal of Statistical Software, 16 Sept. 2010, www.jstatsoft.org/article/view/v036i11. 

**Dataset**
https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset/viewer/default/train?views%5B%5D=train&row=10

Aclanthology. (n.d.). https://aclanthology.org/W13-1728.pdf 