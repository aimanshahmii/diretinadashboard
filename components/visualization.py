import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_visualization_section():
    """
    Create the visualization section with more detailed charts and analysis
    """
    st.title("Visualizations & Analysis")
    st.markdown("### Analyze detection trends and patterns")
    
    # If we have predictions data
    if hasattr(st.session_state, 'prediction_history') and len(st.session_state.prediction_history) > 0:
        # Convert the prediction history to a DataFrame
        df = pd.DataFrame(st.session_state.prediction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add a date column for grouping
        df['date'] = df['timestamp'].dt.date
        
        # Line chart of detections over time
        st.subheader("Myopia Detections Over Time")
        
        # Group by date and diagnosis
        date_diagnosis = df.groupby(['date', 'prediction']).size().reset_index(name='count')
        date_diagnosis['Diagnosis'] = date_diagnosis['prediction'].apply(lambda x: 'Myopia' if x == 1 else 'Normal')
        
        # Check if we have enough dates for a meaningful chart
        unique_dates = date_diagnosis['date'].nunique()
        
        if unique_dates > 1:
            # Create a line chart
            fig = px.line(
                date_diagnosis, 
                x='date', 
                y='count', 
                color='Diagnosis',
                color_discrete_map={'Myopia': '#FF6B6B', 'Normal': '#4CAF50'},
                markers=True
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Cases",
                legend_title="Diagnosis"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Not enough dates, show a bar chart instead
            diagnosis_counts = df['prediction'].value_counts().reset_index()
            diagnosis_counts.columns = ['Diagnosis', 'Count']
            diagnosis_counts['Diagnosis'] = diagnosis_counts['Diagnosis'].apply(lambda x: 'Myopia' if x == 1 else 'Normal')
            
            fig = px.bar(
                diagnosis_counts,
                x='Diagnosis',
                y='Count',
                color='Diagnosis',
                color_discrete_map={'Myopia': '#FF6B6B', 'Normal': '#4CAF50'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("More data over multiple days is needed for trend analysis.")
        
        # Confidence distribution
        st.subheader("Prediction Confidence Distribution")
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of confidence scores by diagnosis
            df['Confidence'] = df['confidence']
            df['Diagnosis'] = df['prediction'].apply(lambda x: 'Myopia' if x == 1 else 'Normal')
            
            fig = px.histogram(
                df,
                x='Confidence',
                color='Diagnosis',
                nbins=10,
                range_x=[0, 1],
                color_discrete_map={'Myopia': '#FF6B6B', 'Normal': '#4CAF50'}
            )
            fig.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot of confidence vs time
            fig = px.scatter(
                df,
                x='timestamp',
                y='confidence',
                color='Diagnosis',
                color_discrete_map={'Myopia': '#FF6B6B', 'Normal': '#4CAF50'},
                hover_name='image_name',
                labels={'confidence': 'Confidence Score', 'timestamp': 'Time'}
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Confidence Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Health insights and risks section
        st.subheader("Health Insights")
        
        # Calculate some statistics
        myopia_percentage = (df['prediction'].sum() / len(df)) * 100
        avg_confidence = df['confidence'].mean() * 100
        
        # Create gauges for insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Myopia prevalence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=myopia_percentage,
                title={'text': "Myopia Prevalence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0066CC"},
                    'steps': [
                        {'range': [0, 10], 'color': "#E8F5E9"},
                        {'range': [10, 50], 'color': "#FFFDE7"},
                        {'range': [50, 100], 'color': "#FFEBEE"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide interpretation
            if myopia_percentage < 10:
                st.success("Low myopia prevalence. Continue regular screening.")
            elif myopia_percentage < 50:
                st.warning("Moderate myopia prevalence. Consider increased screening frequency.")
            else:
                st.error("High myopia prevalence. Immediate attention and intervention recommended.")
        
        with col2:
            # Model confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_confidence,
                title={'text': "Average Prediction Confidence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0066CC"},
                    'steps': [
                        {'range': [0, 60], 'color': "#FFEBEE"},
                        {'range': [60, 80], 'color': "#FFFDE7"},
                        {'range': [80, 100], 'color': "#E8F5E9"}
                    ],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide interpretation
            if avg_confidence < 60:
                st.error("Low prediction confidence. Consider acquiring better quality images.")
            elif avg_confidence < 80:
                st.warning("Moderate prediction confidence. Results should be verified by an ophthalmologist.")
            else:
                st.success("High prediction confidence. Results are likely reliable.")
    
    else:
        # Show empty state
        st.info("No data available for visualization. Upload and analyze images to see detailed charts and insights.")
        
        # Display a medical dashboard image
        st.image(
            "https://images.unsplash.com/photo-1514416432279-50fac261c7dd",
            caption="Medical Analysis Dashboard",
            width=600
        )
        
        # Example of what the visualizations will show
        st.markdown("""
        ### What You'll See Here
        
        Once you've analyzed some fundus images, this section will show:
        
        1. **Trend charts** - Visualize myopia detection patterns over time
        2. **Confidence distribution** - See how certain the AI is about its predictions
        3. **Health insights** - Get population-level statistics and risk indicators
        
        Upload images using the 'Upload & Predict' section to start building your visualization dashboard.
        """)
