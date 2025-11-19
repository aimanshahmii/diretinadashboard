import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def create_dashboard():
    """
    Create the main dashboard view with summary statistics and visualizations
    """
    st.title("DiRetina Dashboard")
    st.markdown("### AI-powered Myopia Detection from Fundus Images")
    
    # Show notification on fresh session
    if st.session_state.get('is_fresh_session', False):
        st.info("Dashboard metrics have been reset. Upload new images to see updated statistics.")
        # Reset the flag so it doesn't show again in this session
        st.session_state.is_fresh_session = False
    
    # Top row with key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics from session state
    total_images = len(st.session_state.predictions) if hasattr(st.session_state, 'predictions') else 0
    myopia_count = sum(1 for pred in st.session_state.predictions if pred[0] == 1) if hasattr(st.session_state, 'predictions') else 0
    normal_count = total_images - myopia_count
    
    # Calculate eye health score (example metric: percentage of normal eyes)
    eye_health_score = (normal_count / total_images * 100) if total_images > 0 else 0
    
    # Display metrics in cards
    with col1:
        st.metric(label="Images Analyzed", value=total_images)
    
    with col2:
        st.metric(label="Myopia Detected", value=myopia_count)
    
    with col3:
        st.metric(label="Normal Eyes", value=normal_count)
    
    with col4:
        st.metric(label="Eye Health Score", value=f"{eye_health_score:.1f}%")
    
    # Middle row with visualizations
    st.markdown("### Visualization Dashboard")
    
    # If we have prediction data, show visualizations
    if hasattr(st.session_state, 'predictions') and len(st.session_state.predictions) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnosis Distribution")
            # Create pie chart
            fig = px.pie(
                names=['Myopia', 'Normal'],
                values=[myopia_count, normal_count],
                color=['Myopia', 'Normal'],
                color_discrete_map={'Myopia': '#FF6B6B', 'Normal': '#4CAF50'},
                hole=0.4,
            )
            fig.update_layout(
                legend_title="Diagnosis",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Confidence")
            
            # Get confidence values for myopia and normal cases
            myopia_confidences = [pred[1] * 100 for pred in st.session_state.predictions if pred[0] == 1]
            normal_confidences = [pred[1] * 100 for pred in st.session_state.predictions if pred[0] == 0]
            
            # Create box plot
            fig = go.Figure()
            
            if myopia_confidences:
                fig.add_trace(go.Box(y=myopia_confidences, name="Myopia", marker_color='#FF6B6B'))
            
            if normal_confidences:
                fig.add_trace(go.Box(y=normal_confidences, name="Normal", marker_color='#4CAF50'))
                
            fig.update_layout(
                yaxis_title="Confidence (%)",
                boxmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Bottom row with additional information
        st.markdown("### Recent Activity")
        
        # Create a dataframe for the prediction history
        if hasattr(st.session_state, 'prediction_history') and len(st.session_state.prediction_history) > 0:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['Diagnosis'] = history_df['prediction'].apply(lambda x: 'Myopia' if x == 1 else 'Normal')
            history_df['Confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            
            # Show the recent predictions
            display_df = history_df[['timestamp', 'image_name', 'Diagnosis', 'Confidence']].rename(
                columns={'timestamp': 'Time', 'image_name': 'Image'}
            )
            st.dataframe(display_df.sort_values('Time', ascending=False))
        else:
            st.info("No prediction history available. Upload images to start analyzing.")
    else:
        # Show empty state with instructions
        st.info(
            "No data available yet. Upload fundus images using the 'Upload & Predict' section "
            "to see visualizations and statistics."
        )
