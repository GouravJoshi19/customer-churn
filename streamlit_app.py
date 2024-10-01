import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset

data=pd.read_csv("./new_data")
data=data.drop(columns='Unnamed: 0')


st.title("ChurnMaster: Predicting Customer Churn with Machine Learning")


# Display dataset description
st.write("# Bank Churn Dataset")
st.write("This dataset contains information about customers and their churn status.")
st.write("Number of samples:", data.shape[0])
st.write("Number of features:", data.shape[1])
st.write("## Features:")
for column in data.columns:
    st.write("- **{}:** {}".format(column, " ".join(data[column].dropna().sample(2).astype(str).tolist())))


# Summary statistics
st.write("## Summary Statistics")
st.write(data.describe())

# Age distribution
st.write("## Age Distribution")
histogram=sns.histplot(data=data, x='Age', bins=30, kde=True)
plt.title("Age distribution in the Dataset")
plt.legend()
plt.show()
st.pyplot()


#age-churn
churned = data[data['Exited'] == 1]['Age']
not_churned = data[data['Exited'] == 0]['Age']

# Create histogram
plt.hist([churned, not_churned], bins=20, color=['red', 'blue'], label=['Churned', 'Not Churned'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Ages for Churned and Non-Churned Customers')
plt.legend()
plt.show()
st.pyplot()

# Filter by gender
gender = st.selectbox("Select Gender", options=["All"] + data['Gender'].unique().tolist())
if gender != "All":
    filtered_data = data[data['Gender'] == gender]
    st.write("## Summary Statistics for", gender, "Customers")
    st.write(filtered_data.describe())


#Gender Distribution
st.write("## Gender Distribution")
fig=sns.countplot(data=data,x='Gender')
plt.title("Gender distribution in the Dataset")
plt.legend()
plt.show()
st.pyplot()

#Proportion-gender
churned_gender=data[data['Exited']==1]["Gender"]
not_churned_gender=data[data['Exited']==0]["Gender"]
histogram_3=plt.hist([churned_gender,not_churned_gender],color=['green','yellow'],label=['Churned','Not-Churned'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Genders for Churned and Non-Churned Customers')
plt.legend()
plt.show()
st.pyplot()


st.set_option('deprecation.showPyplotGlobalUse', False)

# Churn proportion
churn_proportion = data['Exited'].value_counts(normalize=True)
st.write("## Churn Proportion")
st.write(churn_proportion)


model = pickle.load(open("./model.pkl","rb"))
def encoder(string):
    if(string=="Yes"):
        return 1
    else:
        return 0

st.sidebar.title("Input Features")    
def input_features():
    credit_score=st.sidebar.slider("CreditSCore",min_value=float(data['CreditScore'].min()),max_value=float(data['CreditScore'].max()),value=float(data['CreditScore'].mean()))
    age=st.sidebar.slider("Age",min_value=float(data['Age'].min()),max_value=float(data['Age'].max()),value=float(data['Age'].mean()))
    tenure=st.sidebar.slider("Tenure",min_value=float(data['Tenure'].min()),max_value=float(data['Tenure'].max()),value=float(data['Tenure'].mean()))
    balance=st.sidebar.slider("Balance",min_value=float(data['Balance'].min()),max_value=float(data['Balance'].max()),value=float(data['Balance'].mean()))
    salary=st.sidebar.slider("Salary",min_value=float(10000),max_value=float(200000),value=float(data['EstimatedSalary'].mean()))
    products=st.sidebar.selectbox("NumOfProducts",options=data['NumOfProducts'].sort_values().unique(),index=None)
    crcard=st.sidebar.selectbox("HasCrCard",options=['Yes','NO'],index=None)
    activemember=st.sidebar.selectbox("IsActiveMember",options=['Yes','NO'],index=None)
    gender=st.sidebar.selectbox("Gender",options=['Male','Female'],index=None)
    country=st.sidebar.selectbox("Country",options=['France','Spain','Germany'],index=None)
    Data={
     'CreditScore':credit_score,
     'Age':age,
     'Tenure':tenure,
     'Balance':balance,
     'EstimatedSalary':salary,
     'NumOfProducts':products,
     'HasCrCard':encoder(crcard),
     'IsActiveMember':encoder(activemember),
     'Gender':gender,
     'Geography':country}
    features=pd.DataFrame(Data,index=[0])
    return features  

input_data=input_features()

res=''
st.markdown('''-----''')
st.write('## Predicter')
st.write("To predict that whether the customer Has Stayed or not .\nYou can use the input features provided in the sidebar")
input_data.dropna(inplace=True)
try:
    if st.button("Predict"):
        res=model.predict([input_data][0])
        if(res==0):
            st.success("The customer Has Stayed")
        elif(res==1):
            st.warning("The customer Has Exited")
except ValueError as ve:
    st.warning("Please check all the values")
st.write("---")
st.write("# Conclusion")

st.write("This web app provides an overview of the bank churn dataset and explores various aspects of customer churn.")
st.write("Based on the analysis, it appears that there is a significant proportion of churned customers, particularly among younger age groups.")


# Footer
st.write("---")
st.write("Created  by Gourav Joshi")

