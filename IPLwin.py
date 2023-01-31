import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

column1, column2 = st.columns(2)

with column1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with column2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

column3, column4, column5 = st.columns(3)

with column3:
    score = st.number_input('Score')
with column4:
    overs = st.number_input('Overs completed')
with column5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_to_get = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    curr_runrate = score / overs
    req_runrate = (runs_to_get * 6) / balls_left

    data = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                         'runs_to_get': [runs_to_get], 'balls_left': [balls_left], 'wickets': [wickets],
                         'total_runs_x': [target], 'curr_runrate': [curr_runrate], 'req_runrate': [req_runrate]})

    result = pipe.predict_proba(data)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
