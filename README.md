Python Code

[Data cleaning code](https://github.com/lindaxie7/Income-Prediction/blob/main/data_cleaning.ipynb)

[Evaluate dataset with different machine leanring models code](https://github.com/lindaxie7/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)

[Live prediction with Logistic Regression code](https://github.com/lindaxie7/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)

## Results
Results here: https://incomeclassifier.herokuapp.com/

### Project Overview:
We implemented a web application to make live predictions on one's income greater or less than 50,000K with ten features as inputs. Datasets collected through a public website then performed data cleaning and processing, performed data analysis using SVM, Logistic Regression, and Random Forest algorithms, and then picked Logistic Regression machine learning model for the live prediction. Used Python, Flask, Javascript, HTML for implementation, and Tableau for visualization.


###  Data Source:
Our datasource comes from kaggle: https://www.kaggle.com/lodetomasi1995/income-classification


### Data-processing
1.	Find and drop total of 2399 missing data. Remaining 30162 rows of data for our machine learning model.
2.	Drop unnecessary columns: "fnlwgt", "education-num", "capital-gain", "capital-loss".
3.	Using `One-Hot Encoding` to fit and transform categorical variables.
4.	Using `StandardScaler` module standardizes the data so that the mean of each feature is 0 and standard deviation is 1.

### Features used in the model
- Age: Current age (in years)
- Workclass: Employment status of an individual (e.g.	Private)
- Education: Highest degree obtained (e.g. HS-grad, Bachelors)
- Marital-status: Marital status of an individual (e.g. Divorced, Married-AF-spouse)
- Occupation: Industry/role employed in (e.g. Exec-managerial, Craft-repair)
- Relationship: Represents what this individual is relative to others(e.g. Husband, Wife)
- Race: Descriptions of an individual’s race (e.g. White, Black)
- Sex: Gender assigned at birth (Male/Female)
- Hours-per-week: Weekly hours employed (e.g. 40)
- Native-country: Country of origin for an individual


## Tools/technologies for the creating the dashboard

Programming: (Javascript, Python, HTML):

* Running python script to grab inputs from web entry running ML model (logistic regression) to predict income level based on input, with output showing whether user is likely to have income above or below $50K per year. Additional pages with deeper analysis and background, as described in interactive elements.

* Tableau public: Summarizing trends in data set among features used


