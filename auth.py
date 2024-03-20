import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    # Load the data
    def load_data():
        return pd.read_csv('event_trigger_data_capacity_per_downtime.csv')

    data = load_data()

    # Visualizations
    st.title('System Events Visualization')

    # Convert Event Type to descriptive labels
    data['Event Type'] = data['Event Type'].map({1: 'running', 0: 'not running'})

    # Create bar graph for Event Type counts using Plotly
    fig_bar = px.bar(data['Event Type'].value_counts(), 
                    labels={'value': 'Count', 'index': 'Event Type'}, 
                    title='Count of Running vs. Not Running Events',
                    height=500,
                    color_discrete_map={'running': 'blue', 'not running': 'red'})  

    # Add hover information
    fig_bar.update_traces(hovertemplate='%{x}: %{y}')

    # Create pie chart for reasons
    fig_pie = px.pie(data['Reason'].value_counts(), 
                    names=data['Reason'].value_counts().index,
                    title='Distribution of Reasons for Events')

    # Create line chart for Capacity over Timestamp
    fig_line = px.line(data, x='Timestamp', y='Capacity', title='Capacity over Time')

    
    st.plotly_chart(fig_bar)
    st.plotly_chart(fig_pie)
    st.plotly_chart(fig_line)

    fig = px.bar(data['Operator'].value_counts().reset_index(), 
             x='index', 
             y='Operator', 
             labels={'index': 'Operator', 'Operator': 'Frequency'},
             title='Frequency of Each Operator')

    # Show the graph using Streamlit
    st.plotly_chart(fig)
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

