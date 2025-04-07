 Project Title: Wine Dataset Classification with k-NN (From Scratch)

 Project Description
In this project, we use the Wine dataset from the UCI Machine Learning Repository to solve a classification problem.
 The goal is to classify wines into 3 different classes (Class 1, 2, or 3) based on 13 chemical features.
 The main objective is to implement the k-Nearest Neighbors (k-NN) algorithm from scratch (without using scikit-learn’s KNeighborsClassifier),
 test it with different values of K, and compare the performance of two distance metrics: Euclidean and Manhattan.

 Main Objectives
- Data preprocessing: check for missing values, normalize features
- Split data into training and testing sets (80% / 20%)
- Implement the k-NN algorithm manually using Python
- Evaluate model performance for K = 1, 3, 5, 7, 9
- Compare distance metrics: Euclidean vs. Manhattan
- Plot Accuracy vs. K for both metrics
- Generate confusion matrix and classification report for the best K

 Tools & Libraries Used
- Python (NumPy, Pandas, Matplotlib, Seaborn)
- scikit-learn (only for accuracy_score, confusion_matrix, classification_report)
- Google Colab or Jupyter Notebook environment

 Implementation Instructions

1. Download the dataset from:  
   https://archive.ics.uci.edu/dataset/109/wine  
   (Files inside: wine.data, wine.names, Index)

2. Load the dataset into a Pandas DataFrame.

3. Assign column names based on wine.names file.

4. Check for missing values using isnull().

5. Normalize the features using StandardScaler (z-score normalization).

6. Split the data using train_test_split (test_size=0.2, stratify=y).

7. Write a custom k-NN classifier function with support for:
   - Euclidean distance
   - Manhattan distance

8. Predict on the test set for K = 1, 3, 5, 7, 9 and calculate accuracy.

9. Plot Accuracy vs K for each distance metric.

10. Select the best model and generate:
    - Confusion matrix
    - Classification report

 Final Result
- Best performance: Euclidean distance with K = 7
- Accuracy: 100%
- All classes were correctly classified (perfect precision, recall, and F1-score)

---

You can now use this markdown file as documentation or include it at the top of your Colab notebook or GitHub repo.