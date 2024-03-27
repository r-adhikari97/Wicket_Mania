import streamlit as st
import pandas as pd
import pickle

# loading Model
model = pickle.load(open('model/cricket_model.pkl','rb'))


# Set title
st.title("Wicket Wizard: IPL Cricket Match Predictor")

# Data
teams = ['Sunrisers Hyderabad',
          'Mumbai Indians',
          'Royal Challengers Bangalore', 
          'Kolkata Knight Riders', 
          'Kings XI Punjab',
          'Chennai Super Kings', 
          'Rajasthan Royals',
          'Delhi Capitals'
        ]

city = [
        'Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
        'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
        'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
        'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
        'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
        'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',
        'Mohali', 'Bengaluru']

# Setting Up Columns onto Page
col1, col2 = st.columns(2)

with col1:
    batting_team= st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team= st.selectbox('Select the bowling team', sorted(teams))


# Get City
city = st.selectbox("Select your City", sorted(city))

# Get target
target = st.number_input('Target')

# Setting up 3 columns 
col3, col4, col5 = st.columns(3)


# Setting up Columns
with col3:
    score = st.number_input('Score')

with col4:
    over = st.number_input('Overs Dome')

with col3:
    wicket = st.number_input('Wickets out')

if st.button('Predict Probablity'):
    runs_left = target - score 
    balls_left = 120 - (over*6)
    wicket = 10 - wicket
    crr = score / over
    rrr = (runs_left*6) / balls_left

    input_df = pd.DataFrame(
        {
            "batting_team":[batting_team],
            "bowling_team":[bowling_team],
            "city":[city],
            "runs_left":[runs_left],
            "balls_left":[balls_left],
            "total_runs_x":[target],
            "wicket":[wicket],
            "CRR":[crr],
            "RRR":[rrr]
        }
    )

    # Predictions using Model
    result = model.predict_proba(input_df)

    # Win-Loss Probablity
    loss = result[0][0]
    win = result[0][1]

    # Loading Results
    st.header(batting_team + " - "+ str(round(win*100))+"%")
    st.header(bowling_team + " - "+ str(round(loss*100))+"%")



