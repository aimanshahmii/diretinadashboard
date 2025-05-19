import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
from components.dashboard import create_dashboard
from components.upload import create_upload_section
from components.visualization import create_visualization_section
from components.database_mgmt import create_database_mgmt_section
from components.heatmap_view import create_heatmap_view
from utils.model import load_model
from utils.image_processing import preprocess_image
from utils.local_database import (
    add_local_prediction as add_prediction,
    check_local_db_connection as check_db_connection
)

# Import the local database functions
from utils.local_database import (
    init_local_db,
    get_all_local_predictions, 
    get_all_local_patients,
    add_local_patient
)

# Override the database functions in the database.py
import utils.database
utils.database.init_db = init_local_db
utils.database.get_all_predictions = get_all_local_predictions
utils.database.get_all_patients = get_all_local_patients
utils.database.add_patient = add_local_patient

# Set the SQLite database file path
# Modify this to change the location of your SQLite database file
SQLITE_DB_PATH = "diretina.db"

# Page configuration
st.set_page_config(
    page_title="DiRetina Dashboard (Local SQLite)",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
# Add version tracking to detect app restarts
if "app_version" not in st.session_state:
    # This is a new session or app restart, reset all session variables
    st.session_state.app_version = "1.0"
    st.session_state.uploaded_images = []
    st.session_state.predictions = []
    st.session_state.prediction_history = []
    st.session_state.model = None
    st.session_state.last_upload_time = None
    st.session_state.show_db_config = False
    st.session_state.db_url = None
    st.session_state.is_fresh_session = True
    st.session_state.processed_files = set()  # Reset processed files on restart
else:
    # Existing session, just make sure all variables exist
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "last_upload_time" not in st.session_state:
        st.session_state.last_upload_time = None
    if "show_db_config" not in st.session_state:
        st.session_state.show_db_config = False
    if "db_url" not in st.session_state:
        st.session_state.db_url = None
    if "is_fresh_session" not in st.session_state:
        st.session_state.is_fresh_session = False
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

# Load model
@st.cache_resource
def get_model():
    return load_model()

# Main app layout
def main():
    # Sidebar
    with st.sidebar:
        st.title("DiRetina Dashboard")
        st.markdown("### Navigation")
        page = st.radio("Go to", ["Dashboard", "Upload & Predict", "Visualizations", "Heatmap Analysis", "Database Management"])
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "DiRetina uses AI to detect myopia and other eye diseases from fundus images. "
            "Upload images to get predictions and visualize trends."
        )
        
        # Show local database info
        st.markdown("---")
        st.markdown("### Local Database")
        st.info(f"Using SQLite database: {SQLITE_DB_PATH}")
        
        # Display sample fundus images
        with st.expander("Sample Fundus Images"):
            col1, col2 = st.columns(2)
            with col1:
                st.image("https://images.unsplash.com/photo-1635012641245-0adf659dac97", 
                        caption="Normal Fundus", width=120)
                st.image("https://images.unsplash.com/photo-1635012861357-4742bd877c0c", 
                        caption="Myopia Fundus", width=120)
            with col2:
                st.image("https://images.unsplash.com/photo-1634982839177-9ccc2cf896c5", 
                        caption="Normal Fundus", width=120)
                st.image("https://images.unsplash.com/photo-1634983453360-341a308a5f38", 
                        caption="Myopia Fundus", width=120)
    
    # Make sure model is loaded
    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model = get_model()
    
    # Main content based on selected page
    if page == "Dashboard":
        create_dashboard()
    
    elif page == "Upload & Predict":
        create_upload_section()
    
    elif page == "Visualizations":
        create_visualization_section()
    
    elif page == "Heatmap Analysis":
        create_heatmap_view()
        
    elif page == "Database Management":
        create_database_mgmt_section()
    


if __name__ == "__main__":
    main()