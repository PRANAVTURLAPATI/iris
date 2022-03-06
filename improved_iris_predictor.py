# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'improved_iris_app.py'.
# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
ObjectSVC = SVC(kernel = "linear")
ObjectSVC.fit(x_train, y_train)
# Creating a Logistic Regression model. 
ObjectLG = LogisticRegression(n_jobs = -1)
ObjectLG.fit(x_train, y_train)
# Creating a Random Forest Classifier model.
ObjectRFC = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
ObjectRFC.fit(x_train, y_train)

# S2.2: Copy the following code in the 'improved_iris_app.py' file after the previous code.
# Create a function that accepts an ML mode object say 'model' and the four features of an Iris flower as inputs and returns its name.
def prediction(model, s_length, s_width, p_length, p_width):
  if model.predict([[s_length, s_width, p_length, p_width]])[0] == 0:
    return "Iris Setosa"
  elif model.predict([[s_length, s_width, p_length, p_width]])[0] == 1:
    return "Iris Virginica"
  else:
    return "Iris Versicolor"

# S2.3: Copy the following code and paste it in the 'improved_iris_app.py' file after the previous code.
# Add title widget
st.sidebar.title("Iris Flower Species Prediction App")
# Add 4 sliders and store the value returned by them in 4 separate variables. 
# The 'float()' function converts the 'numpy.float' values to Python float values.
sepal_length_input = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
sepal_width_input = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
petal_length_input = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
petal_width_input = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))
# Add a select box in the sidebar with the 'Classifier' label.
# Also pass 3 options as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in the 'classifier' variable.
classifier = st.sidebar.selectbox("Classifier: ", ("Support Vector Machine", "Logistic Regression", "Random Forest Classifier"))
# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button("Predict"):
  if classifier == "Support Vector Machine":
    flower_species = prediction(ObjectSVC, sepal_length_input, sepal_width_input, petal_length_input, petal_width_input)
    flower_score = ObjectSVC.score(x_train, y_train)
  elif classifier == "Logistic Regression":
    flower_species = prediction(ObjectLG, sepal_length_input, sepal_width_input, petal_length_input, petal_width_input)
    flower_score = ObjectLG.score(x_train, y_train)
  else:
    flower_species = prediction(ObjectRFC, sepal_length_input, sepal_width_input, petal_length_input, petal_width_input)
    flower_score = ObjectRFC.score(x_train, y_train)
  st.write(f"Species Predicted: {flower_species}\nScore: {flower_score}")