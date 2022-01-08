#!/usr/bin/env python
# coding: utf-8

# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from pickle import dump, load
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Initialize the flask App
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# Load the model from its pickle file. (This pickle 
# file was originally saved by the code that trained 
# the model. See mlmodel.py)
LogisticRegression = load(open('pkl/logisticregression_new.pkl', 'rb'))

# Load the scaler from its pickle file. (This pickle
# file was originally saved by the code that trained 
# the model. See mlmodel.py)
scaler = load(open('pkl/scaler_new.pkl','rb'))

# Define the index route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def return_home():
    return render_template('index.html')

# Define the about and credits route
@app.route('/about')
def about():
    return render_template('credits.html')

# Define the svm route
@app.route('/svm')
def svm():
    return render_template('svm.html')

# Define the about and credits route
@app.route('/randomforest')
def random_forest():
    return render_template('random_forest.html')

# Define the about and credits route
@app.route('/logregression')
def log_regression():
    return render_template('logistic_reg.html')

# Define the about and credits route
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# Define a route that runs when the user clicks the Predict button in the web-app
@app.route('/predict',methods=['POST'])
def predict():
    
    # Create a list of the output labels.
    prediction_labels = ['>$50K', '<=$50K']
    
    # Read the list of user-entered values from the website. Note that these will be strings. 
    # features = [x for x in request.form.values()]
    # features = json.loads(request.data)
   
    features = request.json
    print(f"Print thefeatures:{features}")
    # return str(features)

    # print(features["age"],features["workclass"],features["degree_type"],features["maritalstatus"],features["occupationType"],features["relationship"],features["race"],
    #     features["gender"],features["hoursPerWeek"],features["nativecountry"])

    value_arr = [[features["age"],features["workclass"],features["degree_type"],features["maritalstatus"],features["occupationType"],features["relationship"],features["race"],
        features["gender"],features["hoursPerWeek"],features["nativecountry"]]]

    value_df = pd.DataFrame(value_arr, columns=["age", "workclass", "education", "marital-status", "occupation", "relationship",  "race", "sex", "hours-per-week", "native-country"]
                        )
    print(f"Print value_df:{value_df}")

        # Transform category columns
    def changeWorkclass(workclass):
        if workclass == "Private":
            return 0
        elif workclass == "Self-emp-not-inc":
            return 1
        elif workclass == "Local-gov":
            return 2
        elif workclass == "State-gov":
            return 3
        elif workclass == "Self-emp-inc":
            return 4
        elif workclass == "Federal-gov":
            return 5
        elif workclass == "Without-pay":
            return 6
        else:
            return 999
        
    def changeEducation(education):
        if education == "Preschool":
            return 0
        elif education == "1st-4th":
            return 1
        elif education == "5th-6th":
            return 2
        elif education == "7th-8th":
            return 3
        elif education == "9th":
            return 4
        elif education == "10th":
            return 5
        elif education == "11th":
            return 6
        elif education == "12th":
            return 7
        elif education == "HS-grad":
            return 8
        elif education == "Some-college":
            return 9
        elif education == "Assoc-voc":
            return 10
        elif education == "Assoc-acdm":
            return 11
        elif education == "Bachelors":
            return 12
        elif education == "Masters":
            return 13
        elif education == "Prof-school":
            return 14
        elif education == "Doctorate":
            return 15
        else:
            return 999

    def changeMarital(marital):
        if marital == "Divorced":
            return 0
        elif marital == "Married-AF-spouse":
            return 1
        elif marital == "Married-civ-spouse":
            return 2
        elif marital == "Married-spouse-absent":
            return 3
        elif marital == "Never-married":
            return 4
        elif marital == "Separated":
            return 5
        elif marital == "Widowed":
            return 6
        else:
            return 999

    def changeOccupation(occupation):
        if occupation == "Adm-clerical":
            return 0
        elif occupation == "Armed-Forces":
            return 1
        elif occupation == "Craft-repair":
            return 2
        elif occupation == "Exec-managerial":
            return 3
        elif occupation == "Farming-fishing":
            return 4
        elif occupation == "Handlers-cleaners":
            return 5
        elif occupation == "Machine-op-inspct":
            return 6
        elif occupation == "Other-service":
            return 7
        elif occupation == "Priv-house-serv":
            return 8
        elif occupation == "Prof-specialty":
            return 9
        elif occupation == "Protective-serv":
            return 10
        elif occupation == "Sales":
            return 11
        elif occupation == "Tech-support":
            return 12
        elif occupation == "Transport-moving":
            return 13
        else: 
            return 999
        
    def changeRelationship(relationship):
        if relationship == "Husband":
            return 0
        elif relationship == "Wife":
            return 1
        elif relationship == "Not-in-family":
            return 2
        elif relationship == "Own-child":
            return 3
        elif relationship == "Unmarried":
            return 4
        elif relationship == "Other-relative":
            return 5
        else:
            return 999

    def changeRace(race):
        if race == "White":
            return 0
        elif race == "Black":
            return 1
        elif race == "Asian-Pac-Islander":
            return 2
        elif race == "Amer-Indian-Eskimo":
            return 3
        elif race == "Other":
            return 4
        else:
            return 999
        
    def changeSex(sex):
        if sex == "Female":
            return 0
        elif sex == "Male":
            return 1
        else:
            return 999

    def changeCountry(country):
        if country == "United-States":
            return 0
        elif country == "Mexico":
            return 1
        elif country == "Philippines":
            return 2
        elif country == "Germany":
            return 3
        elif country == "Puerto-Rico":
            return 4
        elif country == "Canada":
            return 5
        elif country == "El-Salvador":
            return 6
        elif country == "India":
            return 7
        elif country == "Cuba":
            return 8
        elif country == "England":
            return 9
        elif country == "Jamaica":
            return 10
        elif country == "South":
            return 11
        elif country == "China":
            return 12
        elif country == "Italy":
            return 13
        elif country == "Dominican-Republic":
            return 14
        elif country == "Vietnam":
            return 15
        elif country == "Guatemala":
            return 16
        elif country == "Japan":
            return 17
        elif country == "Poland":
            return 18
        elif country == "Columbia":
            return 19
        elif country == "Haiti":
            return 20
        elif country == "Taiwan":
            return 21
        elif country == "Iran":
            return 22
        elif country == "Portugal":
            return 23
        elif country == "Nicaragua":
            return 24
        elif country == "Peru":
            return 25
        elif country == "Greece":
            return 26
        elif country == "France":
            return 27
        elif country == "Ecuador":
            return 28
        elif country == "Ireland":
            return 29
        elif country == "Hong":
            return 30
        elif country == "Cambodia":
            return 31
        elif country == "Trinadad&Tobago":
            return 32
        elif country == "Laos":
            return 33
        elif country == "Thailand":
            return 34
        elif country == "Yugoslavia":
            return 35
        elif country == "Outlying-US(Guam-USVI-etc)":
            return 36
        elif country == "Hungary":
            return 37
        elif country == "Honduras":
            return 38
        elif country == "Scotland":
            return 39
        elif country == "Holand-Netherlands":
            return 40
        else:
            return 999

    value_df["workclass"] = value_df["workclass"].apply(changeWorkclass)
    value_df["education"] = value_df["education"].apply(changeEducation)
    value_df["marital-status"] = value_df["marital-status"].apply(changeMarital)
    value_df["occupation"] = value_df["occupation"].apply(changeOccupation)
    value_df["relationship"] = value_df["relationship"].apply(changeRelationship)
    value_df["race"] = value_df["race"].apply(changeRace)
    value_df["sex"] = value_df["sex"].apply(changeSex)
    value_df["native-country"] = value_df["native-country"].apply(changeCountry)

    print(f"Print encoded value_df:{value_df}")


    # Transform each input using the scaler function.
    value_df_scaled = scaler.transform(value_df)
    print(f"printing input row1 scaled: {value_df_scaled}")
    # Make a prediction for each input.
    predict = LogisticRegression.predict(value_df_scaled)

    return str(predict[0])



# Allow the Flask app to launch from the command line
if __name__ == "__main__":
    app.run(debug=True)