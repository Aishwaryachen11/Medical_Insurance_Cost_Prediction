
## **MEDICAL INSURANCE COST PREDICTION: A COMPARATIVE ANALYSIS USING LINEAR REGRESSION AND GRADIENT REGRESSOR**

### **1. Introduction**
Medical insurance costs are a critical factor in healthcare planning, both for individuals and insurance providers. Predicting these costs accurately allows insurance companies to price their policies more effectively and helps individuals understand their potential medical expenses. In this project, we explore two machine learning approaches—Linear Regression and Gradient Boosting Regressor—to predict medical insurance costs based on various personal attributes. We perform data analysis, feature engineering, model training, and evaluation to compare the performance of these models.

**Linear Regression**

**Concept:**
Linear Regression is a fundamental algorithm in machine learning and statistics used for predicting a continuous dependent variable based on one or more independent variables. It assumes a linear relationship between the input variables (features) and the output variable (target). The goal is to find the line (or hyperplane in the case of multiple variables) that best fits the data.

**Algorithm:**

Initialize: Start with an initial guess for the model parameters (coefficients).
Calculate Predicted Values: Compute the predicted value using the formula:

![image](https://github.com/user-attachments/assets/4807839f-20f1-4cfc-b2cb-4f88c9228549)

where 𝛽0 is the intercept and 𝛽1,…,𝛽𝑛 are the coefficients for each feature 𝑥1,…,𝑥𝑛

**Calculate the Cost Function:** The cost function, typically the Mean Squared Error (MSE), is calculated as:

![image](https://github.com/user-attachments/assets/def8e16b-d17e-4130-a4c1-10d5d14f731e)

where 𝑚 is the number of training examples.
**Optimize:** Adjust the parameters 𝛽 to minimize the cost function using methods like Gradient Descent.
Iterate: Repeat the optimization step until convergence (i.e., when changes in the cost function become negligible).

**Application:**
Linear Regression is best suited for problems where the relationship between the input variables and the output is approximately linear. It is easy to interpret and implement, making it a popular choice for many regression tasks. However, it may not perform well if the underlying data has complex, non-linear relationships.

**Gradient Boosting Regressor**

**Concept:**
Gradient Boosting is an ensemble learning technique that builds models sequentially, with each new model correcting the errors made by the previous ones. The Gradient Boosting Regressor specifically uses decision trees as weak learners to create a strong predictive model. It works by iteratively minimizing the loss function (often the squared error in regression tasks) using gradient descent, which allows the model to learn complex patterns in the data.

**Algorithm:**

Initialize: Start with an initial prediction, often the mean of the target values:

![image](https://github.com/user-attachments/assets/760876c6-56d2-46d8-8cf2-4bceca0829e4)

Iterate: For each tree 𝑡 in the sequence:
Compute Residuals: Calculate the residuals, which represent the difference between the actual values and the predictions from the previous model:

![image](https://github.com/user-attachments/assets/1d8a02ca-9b0f-47ec-969c-e96da960f593)

Fit a New Model: Fit a new decision tree to the residuals. This tree learns the errors made by the previous model.
Update the Prediction: Update the prediction by adding the new tree's predictions, scaled by a learning rate 𝜂

![image](https://github.com/user-attachments/assets/dee675d9-8676-485d-81e7-7dc7f9cab53a)

where 𝑓𝑡(𝑥) is the prediction of the new tree.
Output the Final Model: After a predetermined number of iterations (trees), the final model is a sum of the initial prediction and the predictions from all the trees:

![image](https://github.com/user-attachments/assets/fddf4af2-77ba-4316-9cbf-49a114f252ad)

**Application:**

Gradient Boosting is highly effective in scenarios where the relationships between variables are non-linear and complex. It is widely used in competitions and real-world applications due to its ability to achieve high predictive accuracy. However, it requires careful tuning of parameters such as the number of trees, the depth of each tree, and the learning rate to prevent overfitting and ensure generalization.

**Summary:**

Linear Regression is simple, interpretable, and works well when the relationship between variables is linear. It's easy to implement and provides insights into how each feature impacts the target variable.
Gradient Boosting Regressor is a more complex and powerful algorithm that can model non-linear relationships. It sequentially builds an ensemble of models that collectively improve prediction accuracy. This method is particularly useful when working with complex datasets where interactions between features are not straightforward.

### **2. Objectives of the Project**
The primary objective of this project is to develop an accurate predictive model for medical insurance costs based on demographic and lifestyle factors. This model can help insurance companies set premiums more effectively and provide individuals with a better understanding of their potential medical expenses. To achieve this, the project is divided into several key objectives, including exploratory data analysis (EDA) and model development.

### **3. Dataset Description**
The dataset used was downloaded from Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance

Data preparation: Prepared the data for modeling by handling missing values, outliers, and encoding categorical variables. Addressed data quality issues such as missing values (though in this case, none were found). Encoded categorical variables such as sex, smoker, and region to transform them into a format suitable for machine learning models.

The dataset contains 1,338 records, each representing an individual's information and corresponding medical insurance charges. The features in the dataset include:
•	age: Age of the primary beneficiary.
•	sex: Gender of the insurance contractor (female, male).
•	bmi: Body Mass Index, providing an understanding of body weights that are relatively high or low relative to height.
•	children: Number of children covered by health insurance.
•	smoker: Whether the person is a smoker (yes, no).
•	region: The residential area of the beneficiary in the U.S. (northeast, southeast, southwest, northwest).
•	charges: Individual medical costs billed by health insurance.

3. Data Preparation
3.1 Data Cleaning
The dataset was checked for missing values, and no missing values were found. This ensured that the entire dataset was available for analysis without the need for imputation or removal of records.
3.2 Handling Outliers
Outliers were identified primarily in the bmi and charges columns. Boxplots revealed a few extreme values in the BMI and a significant number of high medical charges. These outliers were retained as they are likely representative of real-world scenarios where some individuals incur much higher medical costs due to specific health conditions.
3.3 Data Encoding
Categorical variables such as sex, smoker, and region were encoded using one-hot encoding. This approach allowed us to convert categorical features into numerical format, suitable for model training.

### **4. Exploratory Data Analysis (EDA)**
Performed exploratory data analysis to uncover patterns, relationships, and insights that will inform the model development process.

1. Analyzed the distribution of key features such as age, BMI, and medical charges to understand their spread and skewness.
2. Computed correlations between the target variable (medical charges) and other features to identify which variables have the strongest relationships with the target.
3. Used visual tools like scatter plots, histograms, and boxplots to visually inspect relationships and potential patterns in the data.
4. Use this link to access with the notebook in Google Colab : [Open Colab Notebook](https://github.com/Aishwaryachen11/Medical_Insurance_Cost_Prediction/blob/main/Medical_Cost_Prediction.ipynb)
   
**4.1 Distribution of BMI and Charges**

<img src="https://github.com/Aishwaryachen11/Medical_Insurance_Cost_Prediction/blob/main/Images/Distribution%20of%20BMI%20and%20Charges.png" alt="Description" width="450"/>

•	BMI: The distribution of BMI was slightly right-skewed, with a mean BMI of 30.66, indicating that the dataset includes individuals with a wide range of body weights.
•	Charges: The distribution of charges was highly right-skewed, with a mean of $13,270.42 and a standard deviation of $12,110. This skewness is expected as medical costs can vary significantly, with a few individuals incurring very high expenses.

**4.2 Correlation Analysis**

<img src="https://github.com/Aishwaryachen11/Medical_Insurance_Cost_Prediction/blob/main/Images/Correlation%20analyis.png" alt="Description" width="450"/>

The correlation between the features and the target variable (charges) was analyzed:
•	smoker_yes: Strong positive correlation with charges (0.79). Smoking status is a significant predictor of higher medical costs.
•	age: Moderate positive correlation with charges (0.30). Older individuals tend to incur higher medical costs.
•	bmi: Weak positive correlation with charges (0.20). Higher BMI is associated with increased medical costs, albeit to a lesser extent.
•	region and children: Weak correlations, indicating minimal direct influence on medical costs.

###**5. Model Development**
Use this link to access with the notebook in Google Colab : [Open Colab Notebook](https://github.com/Aishwaryachen11/Medical_Insurance_Cost_Prediction/blob/main/Medical_Cost_Prediction.ipynb)
**5.1 Linear Regression**
A Linear Regression model was developed to predict medical insurance costs. The model was trained on the training set and evaluated using standard metrics such as MSE, RMSE, MAE, and R-squared (R²).

MSE: 33,596,915.85
RMSE: 5,796.28
MAE: 4,181.19
R²: 0.784

**5.2 Gradient Boosting Regressor**

The Gradient Boosting Regressor model was also trained on the same dataset. This model is an ensemble technique that builds multiple decision trees in a sequential manner to improve prediction accuracy.

MSE: 18,745,176.47
RMSE: 4,329.57
MAE: 2,443.48
R²: 0.879

**6. Model Comparison**
The table below compares the performance of the Linear Regression model and the Gradient Boosting Regressor:

<img src="https://github.com/Aishwaryachen11/Medical_Insurance_Cost_Prediction/blob/main/Images/Model%20Comparision.png" alt="Description" width="450"/>

**Insights from the Comparison Table:**

**Mean Squared Error (MSE):**
The MSE for Gradient Boosting Regressor (18,745,187.00) is significantly lower than that for Linear Regression (33,596,915.85). This suggests that the Gradient Boosting Regressor provides more accurate predictions, with less deviation between the predicted and actual values.

**Root Mean Squared Error (RMSE):**
Similarly, the RMSE for Gradient Boosting Regressor (4,329.57) is lower than that for Linear Regression (5,796.28). This indicates that the Gradient Boosting Regressor has a smaller average prediction error, making it a better model for this dataset.

**Mean Absolute Error (MAE):**
The MAE, which measures the average magnitude of errors in a set of predictions, is also lower for Gradient Boosting Regressor (2,443.48) compared to Linear Regression (4,181.19). This again highlights the superior performance of the Gradient Boosting model in making predictions that are closer to the actual values.

**R-squared (R²):**
The R² value for Gradient Boosting Regressor (0.88) is higher than that for Linear Regression (0.78). This means that the Gradient Boosting Regressor explains 88% of the variance in the medical insurance charges, compared to 78% explained by the Linear Regression model. A higher R² value indicates a better fit of the model to the data.

**Conclusion:**
The Gradient Boosting Regressor outperforms the Linear Regression model across all evaluation metrics, including MSE, RMSE, MAE, and R². This suggests that Gradient Boosting Regressor is better suited for predicting medical insurance costs in this dataset, likely due to its ability to model complex, non-linear relationships in the data. As a result, it provides more accurate and reliable predictions than the simpler Linear Regression model. This insight is crucial for applications where prediction accuracy is paramount, such as in insurance pricing and risk assessment.
