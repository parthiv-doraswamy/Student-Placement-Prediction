# Student Placement Prediction using Machine Learning

## Overview

This project uses Machine Learning to predict whether a student is likely to be placed based on academic performance, internships, projects, aptitude scores, and placement training.

## Dataset Source

The dataset used in this project is the **Placement Prediction Dataset** available on Kaggle.

Dataset Link: https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset

### About the Dataset

This dataset contains information related to students' academic performance, internships, projects, certifications, aptitude scores, training, and placement outcomes. The objective is to predict whether a student will be placed based on these attributes.

### Dataset Features

* CGPA
* Internships
* Projects
* Workshops/Certifications
* Aptitude Test Score
* Soft Skills Rating
* Extracurricular Activities
* Placement Training
* SSC Marks
* HSC Marks

### Target Variable

* Placement Status (Placed / Not Placed)


## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Machine Learning Model

* Logistic Regression

## Project Workflow

1. Loaded and explored the dataset.
2. Performed data preprocessing and label encoding.
3. Split the dataset into training and testing sets.
4. Trained a Logistic Regression model.
5. Evaluated model performance using classification metrics.
6. Generated placement predictions for new student profiles.

## Results

* Dataset Size: 10,000 records
* Model: Logistic Regression
* Accuracy: 79.45%

## Future Improvements

* Experiment with Decision Trees and Random Forests.
* Perform feature importance analysis.
* Build a simple web interface for predictions.
