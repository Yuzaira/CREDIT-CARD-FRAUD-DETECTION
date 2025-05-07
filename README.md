Credit Card Fraud Detection

Project Overview

This project aims to build a machine learning model that can detect fraudulent credit card transactions from highly imbalanced data. Using real-world inspired transaction features, the focus is on correctly identifying fraudulent activities while minimizing false alarms.

⸻

Problem Statement

Financial institutions face significant challenges due to fraudulent transactions. Detecting fraud early is critical but difficult due to:
	•	Class imbalance (fraud is rare)
	•	The cost of false positives (blocking genuine users)
	•	The need for high recall and precision in detection systems

⸻

Dataset
	•	Source: Public credit card transaction dataset
	•	Total transactions: ~285,000
	•	Fraud cases: <1%
	•	Features: Time, Amount, anonymized features (V1-V28), Class (0: non-fraud, 1: fraud)

⸻

Tech Stack
	•	Python
	•	Pandas, NumPy
	•	Matplotlib, Seaborn
	•	Scikit-learn
	•	XGBoost
	•	SMOTE (Imbalanced-learn)

⸻

Steps Followed
	1.	Data Exploration
	•	Class imbalance visualized
	•	Correlation heatmap to detect feature influence
	2.	Preprocessing
	•	Standardized Amount and Time
	•	SMOTE applied to balance the dataset
	3.	Modeling
	•	Logistic Regression
	•	Random Forest Classifier
	•	XGBoost Classifier
	4.	Evaluation
	•	Confusion matrix
	•	Precision, Recall, F1 Score, ROC-AUC
	•	Comparison table for model selection

⸻

Handling Class Imbalance

Since fraudulent transactions make up less than 1% of the data, we used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic fraud samples in the training set and balance both classes.

⸻

Model Performance Comparison
Precision (Fraud)	Recall (Fraud)	F1 Score	ROC-AUC
Model				
LogisticRegression	0.059399	0.928571	0.111656	0.951615
Random Forest	0.892473	0.846939	0.869110	0.923381
XGBoost	0.741071	0.846939	0.790476	0.923214

Final Model
	•	Selected Model: Random Forest Classifier
	•	Reason: Best trade-off between high precision, high recall, and F1-score, making it the most reliable for real-world deployment where minimizing false positives is crucial.

⸻

Limitations
	•	SMOTE creates synthetic data which might not reflect real-world fraud perfectly
	•	Real-time detection and deployment pipeline not included
	•	Hyperparameter tuning not deeply explored

⸻

Future Work
	•	Improve performance with hyperparameter tuning
	•	Try LightGBM or CatBoost
	•	Deploy the model as a Streamlit web app
	•	Use real-time transaction scoring with Flask/Kafka
	•	Include explainability tools like SHAP

⸻

Project Author

Fathima Yusaira
Certified Specialist in Data Science and Analytics
Skills: Python | SQL | Tableau | Excel | Scikit-learn | NLP