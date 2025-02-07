import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

# Preprocess the input data
def preprocess_data(input_dict):
    df = pd.DataFrame([input_dict])

    # Calculate Total Satisfaction
    df['Total_Satisfaction'] = (input_dict['EnvironmentSatisfaction'] +
                                input_dict['JobInvolvement'] +
                                input_dict['JobSatisfaction'] +
                                input_dict['RelationshipSatisfaction'] +
                                input_dict['WorkLifeBalance']) / 5

    # Drop original satisfaction columns
    df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'],
            axis=1, inplace=True)

    # Convert Total satisfaction into boolean
    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
    df.drop('Total_Satisfaction', axis=1, inplace=True)

    # Age boolean
    df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
    df.drop('Age', axis=1, inplace=True)

    # Daily Rate boolean
    df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
    df.drop('DailyRate', axis=1, inplace=True)

    # Department boolean
    df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
    df.drop('Department', axis=1, inplace=True)

    # Distance From Home boolean
    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
    df.drop('DistanceFromHome', axis=1, inplace=True)

    # Job Role boolean
    df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
    df.drop('JobRole', axis=1, inplace=True)

    # Hourly Rate boolean
    df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
    df.drop('HourlyRate', axis=1, inplace=True)

    # Monthly Income boolean
    df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
    df.drop('MonthlyIncome', axis=1, inplace=True)

    # Number of Companies Worked boolean
    df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
    df.drop('NumCompaniesWorked', axis=1, inplace=True)

    # Total Working Years boolean
    df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
    df.drop('TotalWorkingYears', axis=1, inplace=True)

    # Years at Company boolean
    df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsAtCompany', axis=1, inplace=True)

    # Years in Current Role boolean
    df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsInCurrentRole', axis=1, inplace=True)

    # Years Since Last Promotion boolean
    df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

    # Years With Current Manager boolean
    df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsWithCurrManager', axis=1, inplace=True)

    # Business Travel one-hot encoding
    if input_dict['BusinessTravel'] == 'Rarely':
        df['BusinessTravel_Rarely'] = 1
        df['BusinessTravel_Frequently'] = 0
        df['BusinessTravel_No_Travel'] = 0
    elif input_dict['BusinessTravel'] == 'Frequently':
        df['BusinessTravel_Rarely'] = 0
        df['BusinessTravel_Frequently'] = 1
        df['BusinessTravel_No_Travel'] = 0
    else:
        df['BusinessTravel_Rarely'] = 0
        df['BusinessTravel_Frequently'] = 0
        df['BusinessTravel_No_Travel'] = 1
    df.drop('BusinessTravel', axis=1, inplace=True)

    # Education one-hot encoding
    education_columns = {
        1: [1,0,0,0,0],
        2: [0,1,0,0,0],
        3: [0,0,1,0,0],
        4: [0,0,0,1,0],
        5: [0,0,0,0,1]
    }
    edu_cols = education_columns[input_dict['Education']]
    df['Education_1'], df['Education_2'], df['Education_3'], df['Education_4'], df['Education_5'] = edu_cols
    df.drop('Education', axis=1, inplace=True)

    # Education Field one-hot encoding
    education_field_mapping = {
        'Life Sciences': [1,0,0,0,0,0],
        'Medical': [0,1,0,0,0,0],
        'Marketing': [0,0,1,0,0,0],
        'Technical Degree': [0,0,0,1,0,0],
        'Human Resources': [0,0,0,0,1,0],
        'Other': [0,0,0,0,0,1]
    }
    edu_field_cols = education_field_mapping[input_dict['EducationField']]
    (df['EducationField_Life_Sciences'], df['EducationField_Medical'], 
     df['EducationField_Marketing'], df['EducationField_Technical_Degree'], 
     df['Education_Human_Resources'], df['Education_Other']) = edu_field_cols
    df.drop('EducationField', axis=1, inplace=True)

    # Gender one-hot encoding
    df['Gender_Male'] = 1 if input_dict['Gender'] == 'Male' else 0
    df['Gender_Female'] = 1 if input_dict['Gender'] == 'Female' else 0
    df.drop('Gender', axis=1, inplace=True)

    # Marital Status one-hot encoding
    marital_status_mapping = {
        'Married': [1,0,0],
        'Single': [0,1,0],
        'Divorced': [0,0,1]
    }
    marital_cols = marital_status_mapping[input_dict['MaritalStatus']]
    df['MaritalStatus_Married'], df['MaritalStatus_Single'], df['MaritalStatus_Divorced'] = marital_cols
    df.drop('MaritalStatus', axis=1, inplace=True)

    # Overtime one-hot encoding
    df['OverTime_Yes'] = 1 if input_dict['OverTime'] == 'Yes' else 0
    df['OverTime_No'] = 1 if input_dict['OverTime'] == 'No' else 0
    df.drop('OverTime', axis=1, inplace=True)

    # Stock Option Level one-hot encoding
    stock_option_mapping = {
        0: [1,0,0,0],
        1: [0,1,0,0],
        2: [0,0,1,0],
        3: [0,0,0,1]
    }
    stock_cols = stock_option_mapping[input_dict['StockOptionLevel']]
    df['StockOptionLevel_0'], df['StockOptionLevel_1'], df['StockOptionLevel_2'], df['StockOptionLevel_3'] = stock_cols
    df.drop('StockOptionLevel', axis=1, inplace=True)

    # Training Times Last Year one-hot encoding
    training_mapping = {
        0: [1,0,0,0,0,0,0],
        1: [0,1,0,0,0,0,0],
        2: [0,0,1,0,0,0,0],
        3: [0,0,0,1,0,0,0],
        4: [0,0,0,0,1,0,0],
        5: [0,0,0,0,0,1,0],
        6: [0,0,0,0,0,0,1]
    }
    training_cols = training_mapping[input_dict['TrainingTimesLastYear']]
    (df['TrainingTimesLastYear_0'], df['TrainingTimesLastYear_1'], 
     df['TrainingTimesLastYear_2'], df['TrainingTimesLastYear_3'], 
     df['TrainingTimesLastYear_4'], df['TrainingTimesLastYear_5'], 
     df['TrainingTimesLastYear_6']) = training_cols
    df.drop('TrainingTimesLastYear', axis=1, inplace=True)

    return df

def main():
    # Set page title
    st.title('Employee Attrition Prediction')

    # Load the model
    model = load_model()

    # Create input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        business_travel = st.selectbox('Business Travel', ['Rarely', 'Frequently', 'No Travel'])
        daily_rate = st.number_input('Daily Rate', min_value=0, value=500)
        department = st.selectbox('Department', ['Research & Development', 'Sales', 'Human Resources'])
        distance_from_home = st.number_input('Distance From Home', min_value=0, value=5)
        education = st.selectbox('Education', [1, 2, 3, 4, 5])
        education_field = st.selectbox('Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])

    with col2:
        environment_satisfaction = st.slider('Environment Satisfaction', 1, 4, 2)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        hourly_rate = st.number_input('Hourly Rate', min_value=0, value=50)
        job_involvement = st.slider('Job Involvement', 1, 4, 2)
        job_level = st.number_input('Job Level', min_value=1, max_value=5, value=2)
        job_role = st.selectbox('Job Role', ['Laboratory Technician', 'Research Scientist', 'Sales Executive', 'Manager', 'Other'])
        job_satisfaction = st.slider('Job Satisfaction', 1, 4, 2)

    col3, col4 = st.columns(2)

    with col3:
        marital_status = st.selectbox('Marital Status', ['Married', 'Single', 'Divorced'])
        monthly_income = st.number_input('Monthly Income', min_value=0, value=3000)
        num_companies_worked = st.number_input('Number of Companies Worked', min_value=0, value=1)
        overtime = st.selectbox('Overtime', ['Yes', 'No'])
        performance_rating = st.number_input('Performance Rating', min_value=1, max_value=4, value=3)

    with col4:
        relationship_satisfaction = st.slider('Relationship Satisfaction', 1, 4, 2)
        stock_option_level = st.selectbox('Stock Option Level', [0, 1, 2, 3])
        total_working_years = st.number_input('Total Working Years', min_value=0, value=5)
        training_times_last_year = st.selectbox('Training Times Last Year', [0, 1, 2, 3, 4, 5, 6])
        work_life_balance = st.slider('Work Life Balance', 1, 4, 2)

    years_at_company = st.number_input('Years at Company', min_value=0, value=2)
    years_in_current_role = st.number_input('Years in Current Role', min_value=0, value=2)
    years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, value=0)
    years_with_curr_manager = st.number_input('Years With Current Manager', min_value=0, value=2)

    # Prepare input dictionary
    input_dict = {
        'Age': age,
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': overtime,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    # Prediction button
    if st.button('Predict Attrition'):
        # Preprocess the data
        processed_df = preprocess_data(input_dict)
        
        # Make prediction
        prediction = model.predict(processed_df)
        
        # Display result
        if prediction[0] == 0:
            st.success('Prediction: Employee Might Not Leave The Job')
        else:
            st.error('Prediction: Employee Might Leave The Job')

if __name__ == '__main__':
    main()