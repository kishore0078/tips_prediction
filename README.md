📊 Tip Prediction using Linear RegressionThis project builds a Machine Learning pipeline to predict the Tip Amount based on factors like total bill,
party size, sex, smoker status, day, and time using Seaborn's built-in tips dataset.
The pipeline covers end-to-end data processing, feature scaling, encoding, model training, evaluation, and visualization.

🛠️ Requirements & InstallationMake
sure you have Python installed, then install the required dependencies:Bashpip install pandas matplotlib seaborn scikit-learn

⚙️ Workflow & Feature PipelineDataset Loading:
Loads the tips dataset directly via Seaborn.Feature Separation: Separates numerical features (total_bill, size) from categorical features (sex, smoker, day, time).
Categorical Encoding: Applies One-Hot Encoding using pd.get_dummies(drop_first=True) to convert categorical attributes into dummy variables.
Feature Scaling: Normalizes numerical features using StandardScaler for standard score normalization.
Data Splitting: Splits dataset into 80% Training set and 20% Testing set with random_state=42.
Model Training: Trains a Scikit-Learn LinearRegression model.
Evaluation & Visualization:Evaluates performance using the $R^2 Score.
Generates a scatter plot comparing Actual vs. Predicted values against an ideal diagonal reference line.
📈 Model Output & ResultsPerformance Metric: Prints the $R^2 accuracy score as a percentage in the console.
Data Visualization: Displays a scatter plot featuring:X-axis: Actual Tip values (y_test)Y-axis: Predicted Tip values (result)Reference Line: Ideal $y = x$ prediction line in green.

🚀 How to RunSave the code in a fil:
Run the script via command prompt or terminal:Bashpython predict_tips.py
