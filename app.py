import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.utils import to_categorical
import os

# Page config
st.set_page_config(page_title="Traffic Prediction", 
                   page_icon="🚦", 
                   layout="wide")

def classify_traffic(value):
    # Adjusted thresholds based on your dataset
    if isinstance(value, (int, float)):
        if value < 50:
            return "Less Traffic"
        elif value < 90:
            return "Medium Traffic"
        else:
            return "Heavy Traffic"
    return "Unknown"

def predict_traffic(model, data, model_type, scaler):
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    if model_type in ['CNN', 'GRU']:
        # Reshape for deep learning models
        data_reshaped = np.expand_dims(data_scaled, axis=2)
        predictions = model.predict(data_reshaped)
        predicted_classes = np.argmax(predictions, axis=2)[0]
        
        # Map class indices back to traffic conditions
        return [classify_traffic(data[0][i]) for i in range(len(data[0]))]
    else:
        # For traditional ML models
        predictions = model.predict(data_scaled)
        predictions = np.clip(predictions[0], 0, 2)  # Ensure values are in [0, 2]
        return [classify_traffic(val) for val in data[0]]

# Load models and scaler
@st.cache_resource
def load_prediction_models():
    try:
        models = {
            'CNN': load_model('cnn_model.h5'),
            'GRU': load_model('gru_model.h5'),
            'GradientBoosting': joblib.load('gb_model.pkl'),
            'RandomForest': joblib.load('rf_model.pkl')
        }
        scaler = joblib.load('scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Create sample data if not exists
def create_sample_data():
    sample_data = pd.DataFrame({
        'Cross 1': [105,97,76,98,87,80,92,80,91,75,48,61,60],
        'Cross 2': [48,41,47,40,41,40,46,31,55,42,24,31,32],
        'Cross 3': [30,32,44,39,47,35,39,21,35,26,15,17,23],
        'Cross 4': [62,55,58,59,49,63,58,45,46,77,38,33,53],
        'Cross 5': [31,42,40,43,35,34,26,29,26,32,19,16,19],
        'Cross 6': [110,103,100,104,112,89,98,93,89,48,61,54,64]
    })
    sample_data.to_csv('traffic-prediction-dataset.csv', index=False)
    return sample_data

try:
    # Load or create data
    if not os.path.exists('traffic-prediction-dataset.csv'):
        data = create_sample_data()
    else:
        data = pd.read_csv('traffic-prediction-dataset.csv')
    
    # Load models
    models, scaler = load_prediction_models()
    
    if models is None or scaler is None:
        st.error("Failed to load models. Please ensure models are trained.")
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Data Visualization", "Make Prediction"])

    with tab1:
        st.header("Traffic Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Data")
            st.dataframe(data)
            
            # Show class distribution in raw data
            st.subheader("Current Traffic Distribution")
            traffic_conditions = []
            for _, row in data.iterrows():
                conditions = [classify_traffic(val) for val in row]
                traffic_conditions.extend(conditions)
            
            dist_df = pd.DataFrame({
                'Condition': traffic_conditions
            }).value_counts().reset_index()
            dist_df.columns = ['Condition', 'Count']
            
            fig = px.bar(dist_df, x='Condition', y='Count', 
                        title='Distribution of Traffic Conditions')
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Traffic Distribution by Intersection")
            fig = px.box(data)
            st.plotly_chart(fig)
            
            st.subheader("Average Traffic by Intersection")
            avg_traffic = data.mean()
            fig = px.bar(x=data.columns, y=avg_traffic)
            st.plotly_chart(fig)

    with tab2:
        st.header("Predict Traffic")
        
        model_type = st.selectbox(
            "Select Model",
            ['CNN', 'GRU', 'GradientBoosting', 'RandomForest']
        )
        
        input_method = st.radio("Select Input Method", 
                              ["Manual Entry", "CSV Upload"])
        
        if input_method == "Manual Entry":
            st.write("Enter traffic values for each intersection:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cross1 = st.number_input("Cross 1", value=100)
                cross2 = st.number_input("Cross 2", value=50)
            with col2:
                cross3 = st.number_input("Cross 3", value=40)
                cross4 = st.number_input("Cross 4", value=60)
            with col3:
                cross5 = st.number_input("Cross 5", value=35)
                cross6 = st.number_input("Cross 6", value=90)
            
            input_data = np.array([[cross1, cross2, cross3, cross4, 
                                   cross5, cross6]])
            
        else:
            uploaded_file = st.file_uploader("Upload CSV file", 
                                           type=['csv'])
            if uploaded_file is not None:
                input_data = pd.read_csv(uploaded_file).values
            else:
                st.warning("Please upload a CSV file")
                st.stop()
        
        if st.button("Predict Traffic Conditions"):
            # Make predictions
            predictions = predict_traffic(
                models[model_type], input_data, model_type, scaler
            )
            
            # Display results
            st.subheader(f"Prediction Results ({model_type})")
            
            # Create results DataFrame
            results = pd.DataFrame({
                "Intersection": [f"Cross {i+1}" for i in range(6)],
                "Traffic Value": input_data[0],
                "Predicted Condition": predictions
            })
            
            # Style the DataFrame
            def color_conditions(val):
                if val == "Less Traffic":
                    return 'background-color: #90EE90'
                elif val == "Medium Traffic":
                    return 'background-color: #FFD700'
                return 'background-color: #FFB6C1'
            
            st.dataframe(
                results.style.applymap(
                    color_conditions,
                    subset=['Predicted Condition']
                ),
                use_container_width=True
            )
            
            # Visualization
            fig = go.Figure()
            
            for condition in ["Less Traffic", "Medium Traffic", 
                            "Heavy Traffic"]:
                mask = results['Predicted Condition'] == condition
                fig.add_trace(go.Bar(
                    name=condition,
                    x=results[mask]['Intersection'],
                    y=results[mask]['Traffic Value'],
                    marker_color='#90EE90' if condition == "Less Traffic"
                                else '#FFD700' if condition == "Medium Traffic"
                                else '#FFB6C1'
                ))
            
            fig.update_layout(
                title=f"Traffic Predictions ({model_type})",
                xaxis_title="Intersection",
                yaxis_title="Traffic Value",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison
            st.subheader("Model Comparison")
            comparison_results = {}
            
            for model_name, model in models.items():
                model_predictions = predict_traffic(
                    model, input_data, model_name, scaler
                )
                comparison_results[model_name] = model_predictions
            
            comparison_df = pd.DataFrame(comparison_results)
            st.dataframe(comparison_df)

except Exception as e:
    st.error(f"Error: {str(e)}")