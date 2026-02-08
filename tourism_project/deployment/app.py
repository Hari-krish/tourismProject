import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="KarmaLoop/tourism-project", filename="tourism-project-v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Customer Churn Prediction App")
st.write("The Customer Churn Prediction App is an internal tool for bank staff that predicts whether customers are at risk of churning based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")


#Collect User Input
Age = st.number_input("Age of the customer", min_value=18, max_value=60, value=37);
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=1, max_value=5, value=2);
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=1, max_value=21, value=3);
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer", min_value=0, max_value=3, value=1);
MonthlyIncome = st.number_input("Gross monthly income of the customer", min_value=1000, max_value=150000, value=27000);
PitchSatisfactionScore = st.number_input("Score indicating the customer satisfaction with the sales pitch", min_value=1, max_value=5, value=3);
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch", min_value=1, max_value=5, value=3);
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer", min_value=5, max_value=35, value=15);
TypeofContact = st.selectbox("The method by which the customer was contacted ", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("The city category based on development, population, and living standards", ["1", "2", "3"])
Occupation = st.selectbox("Customer's occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
Gender = st.selectbox("Gender of the customer ",['Female', 'Male'])
PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer.",["3","4","5"])
MaritalStatus = st.selectbox("Marital status of the customer",['Single', 'Divorced', 'Married', 'Unmarried'])
Passport = st.selectbox("Whether the customer holds a valid passport",['YES', 'NO'])
OwnCar = st.selectbox("Whether the customer owns a car",['YES','NO'])
Designation = st.selectbox("Customer's designation in their current organization",['Manager', 'Executive', 'Senior Manager', 'VP', 'AVP'])
ProductPitched = st.selectbox("The type of product pitched to the customer",['Deluxe', 'Basic', 'Standard', 'King', 'Super Deluxe'])



# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age                      #Age of the customer.
    'NumberOfPersonVisiting': NumberOfPersonVisiting   #Total number of people accompanying the customer on the trip.
    'NumberOfTrips': NumberOfTrips            #Average number of trips the customer takes annually.
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting #Number of children below age 5 accompanying the customer.
    'MonthlyIncome',: MonthlyIncome           #Gross monthly income of the customer.
    'PitchSatisfactionScore': PitchSatisfactionScore   #Score indicating the customer's satisfaction with the sales pitch.
    'NumberOfFollowups' : PitchSatisfactionScore        #Total number of follow-ups by the salesperson after the sales pitch.
    'DurationOfPitch' : PitchSatisfactionScore          #Duration of the sales pitch delivered to the customer.
    'TypeofContact' : PitchSatisfactionScore          #The method by which the customer was contacted (Company Invited or Self Inquiry).
    'CityTier' : PitchSatisfactionScore               #The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
    'Occupation' : PitchSatisfactionScore             #Customer's occupation (e.g., Salaried, Freelancer).
    'Gender' : PitchSatisfactionScore                 #Gender of the customer (Male, Female).
    'PreferredPropertyStar' : PitchSatisfactionScore  #Preferred hotel rating by the customer.
    'MaritalStatus' : PitchSatisfactionScore         #Marital status of the customer (Single, Married, Divorced).
    'Passport':1 if Passport == "Yes" else 0,               #Whether the customer holds a valid passport (0: No, 1: Yes).
    'OwnCar':1 if OwnCar == "Yes" else 0,				  #Whether the customer owns a car (0: No, 1: Yes).
    'Designation' : Designation           #Customer's designation in their current organization.
    'ProductPitched' : ProductPitched        #The type of product pitched to the customer.
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Prod Not Taken" if prediction == 1 else "Prod Taken"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
