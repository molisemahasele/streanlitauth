import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import statistics

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

st.set_option('deprecation.showPyplotGlobalUse', False)

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

   

    # Preprocessing function
    def preprocess_data(data):
        X = data[['Operator', 'Capacity']]  # Features
        y = data['Event Type']              # Target variable
        return X, y

    # Model training function
    def train_model(X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    # Model evaluation function
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    # Predict function
    def predict_event_type(model, operator_value):
        # Create a DataFrame with the provided operator value
        new_data = pd.DataFrame({'Operator': [operator_value], 'Capacity': [0]})  # Capacity doesn't matter for prediction
        
        # Predict event type
        event_type = model.predict(new_data)
        return "running" if event_type[0] == 1 else "not running"

    # Clustering function
    def perform_clustering(X):
        # Initialize KMeans model
        kmeans = KMeans(n_clusters=2, random_state=42)  # Assuming 2 clusters
        
        # Fit KMeans model
        kmeans.fit(X)
        
        # Assign cluster labels
        cluster_labels = kmeans.labels_
        
        return cluster_labels

    # Plot clustering results
    def plot_clusters(X, cluster_labels):
        plt.figure(figsize=(10, 6))
        plt.scatter(X['Operator'], X['Capacity'], c=cluster_labels, cmap='viridis')
        plt.xlabel('Operator')
        plt.ylabel('Capacity')
        plt.title('Clustering of Operators based on Capacity')
        plt.colorbar(label='Cluster')
        st.pyplot()

    # Streamlit app
    def main():
        st.title("Event Type Prediction and Clustering")
        
        # Load data
        data = load_data()
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Split data for event type prediction
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model for event type prediction
        model = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        
        st.write("Accuracy:", accuracy)
        
        # User input for operator
        operator_value = st.number_input("Enter Operator Value:")

        st.title("Get statistics for capacity")
        
        # Button to calculate statistics
        if st.button("Calculate Capacity Statistics"):
            # Calculate the mean of Capacity
            capacity_mean = data['Capacity'].mean()
            st.write("Mean of Capacity:", capacity_mean)
            
            # Calculate the median of Capacity
            capacity_median = statistics.median(data['Capacity'])
            st.write("Median of Capacity:", capacity_median)
            
            # Calculate the mode of Capacity
            capacity_mode = statistics.mode(data['Capacity'])
            st.write("Mode of Capacity:", capacity_mode)
            
            # Calculate the variance of Capacity
            capacity_variance = data['Capacity'].var()
            st.write("Variance of Capacity:", capacity_variance)
            
            # Calculate the standard deviation of Capacity
            capacity_std_dev = data['Capacity'].std()
            st.write("Standard Deviation of Capacity:", capacity_std_dev)

        
    if __name__ == "__main__":
        main()

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

