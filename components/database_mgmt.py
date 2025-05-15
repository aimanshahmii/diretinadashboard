import streamlit as st
import pandas as pd
from utils.database import (
    check_db_connection, 
    get_all_predictions, 
    get_all_patients,
    add_patient,
    init_db
)
import os

def create_database_mgmt_section():
    """
    Create the database management section
    """
    st.title("Database Management")
    st.markdown("### View and manage database records")
    
    # Initialize database with messages enabled for this section only
    init_db(_show_messages=True)
    
    # Check database connection
    db_connected = check_db_connection(_show_messages=True)
    
    if db_connected:
        st.success("✅ Connected to database")
        
        # Get Database URL from environment
        db_url = os.environ.get('DATABASE_URL', 'Not configured')
        
        # Only show part of the URL for security
        if db_url != 'Not configured':
            # Show only the first part of the URL (hide password)
            parts = db_url.split('@')
            if len(parts) > 1:
                masked_url = f"...@{parts[1]}"
            else:
                masked_url = "Database URL is configured"
            
            st.info(f"Database URL: {masked_url}")
        else:
            st.warning("Database URL not configured. Using in-memory database.")
        
        # Create tabs for different database entities
        tab1, tab2, tab3 = st.tabs(["Predictions", "Patients", "Database Stats"])
        
        with tab1:
            st.subheader("Prediction Records")
            
            # Get all predictions - show messages in this section
            predictions_df = get_all_predictions(_show_messages=True)
            
            if not predictions_df.empty:
                # Format the dataframe for display
                display_df = predictions_df.copy()
                display_df['Diagnosis'] = display_df['prediction'].apply(lambda x: 'Myopia' if x == 1 else 'Normal')
                display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                
                # Display the data
                selected_cols = ['id', 'timestamp', 'image_name', 'Diagnosis', 'Confidence', 'notes']
                display_df = display_df[selected_cols].copy()
                
                # Rename columns for display
                display_df.columns = ['ID', 'Time', 'Image', 'Diagnosis', 'Confidence', 'Notes']
                
                # Sort by time (convert to the right format first)
                try:
                    display_df = display_df.sort_values(by='Time', ascending=False)
                except:
                    # If sorting fails, just continue without sorting
                    pass
                
                # Display
                st.dataframe(display_df)
                
                # Export options
                if st.button("Export Predictions to CSV"):
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="diretina_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No prediction records found in the database.")
                st.markdown("""
                    Upload and analyze images in the "Upload & Predict" section to create prediction records.
                    Once predictions are made, they will be saved to the database automatically.
                """)
        
        with tab2:
            st.subheader("Patient Records")
            
            # Create a form to add new patients
            with st.expander("Add New Patient"):
                with st.form("add_patient_form"):
                    patient_id = st.text_input("Patient ID*", help="Required unique identifier")
                    name = st.text_input("Name", help="Patient name (optional)")
                    age = st.number_input("Age", min_value=0, max_value=120, value=0, help="Patient age (optional)")
                    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], help="Patient gender (optional)")
                    
                    # Form submission
                    submit_button = st.form_submit_button("Add Patient")
                    
                    if submit_button:
                        if not patient_id:
                            st.error("Patient ID is required")
                        else:
                            # Convert age to None if it's 0 (not specified)
                            age_value = age if age > 0 else None
                            # Convert gender to None if it's empty
                            gender_value = gender if gender else None
                            
                            success = add_patient(patient_id, name, age_value, gender_value, _show_messages=True)
                            
                            if success:
                                st.success(f"Patient {patient_id} added successfully")
                                st.rerun()  # Refresh the page to show the new patient
            
            # Get all patients - show messages in this section
            patients_df = get_all_patients(_show_messages=True)
            
            if not patients_df.empty:
                # Display the data
                selected_cols = ['patient_id', 'name', 'age', 'gender', 'created_at']
                display_df = patients_df[selected_cols].copy()
                
                # Rename columns for display
                display_df.columns = ['Patient ID', 'Name', 'Age', 'Gender', 'Created At']
                
                # Display
                st.dataframe(display_df)
                
                # Export options
                if st.button("Export Patients to CSV"):
                    csv = patients_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="diretina_patients.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No patient records found in the database.")
                st.markdown("""
                    Use the "Add New Patient" form above to create patient records.
                """)
        
        with tab3:
            st.subheader("Database Statistics")
            
            # Get counts
            prediction_count = len(get_all_predictions())
            patient_count = len(get_all_patients())
            
            # Create metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Predictions", prediction_count)
            
            with col2:
                st.metric("Total Patients", patient_count)
            
            # Database information
            st.subheader("Database Information")
            
            # Get the database URL (for display only)
            if db_url != 'Not configured':
                db_type = "PostgreSQL (Supabase)" if "postgres" in db_url.lower() else "SQLite (Local)"
            else:
                db_type = "SQLite (In-memory)"
            
            st.info(f"""
                **Database Type:** {db_type}
                
                **Tables:**
                - predictions
                - patients
                
                **Status:** Connected and operational
            """)
    else:
        st.error("❌ Not connected to database")
        
        st.markdown("""
        ### Database Connection Issue
        
        The DiRetina Dashboard is currently unable to connect to a database. 
        
        To set up the database:
        
        1. Create a Supabase project at [supabase.com](https://supabase.com)
        2. Get your database connection string from the Supabase dashboard
        3. Set the `DATABASE_URL` environment variable with the connection string
        
        **For Supabase:**
        1. Go to the Supabase dashboard
        2. Click on your project
        3. Navigate to Project Settings → Database
        4. Find the "Connection string" and select "URI"
        5. Copy the connection string and replace the password placeholder
        
        **Example Connection String:**
        ```
        postgresql://postgres:[YOUR-PASSWORD]@db.example.supabase.co:5432/postgres
        ```
        
        Until a database is configured, the application will use an in-memory database. 
        Note that any data will be lost when the application restarts.
        """)
        
        # Offer to set up the database URL
        if st.button("Configure Database URL"):
            st.session_state.show_db_config = True
        
        if st.session_state.get('show_db_config', False):
            with st.form("db_config_form"):
                db_url = st.text_input(
                    "Database URL", 
                    help="Enter your Supabase PostgreSQL connection string",
                    type="password"
                )
                
                submit = st.form_submit_button("Save Configuration")
                
                if submit and db_url:
                    # In a real application, we would save this to environment variables
                    # or a configuration file. For now, we'll just store it in session state.
                    st.session_state.db_url = db_url
                    st.success("Database URL configured! Please restart the application.")
                    st.info("Note: In this demo, the URL is stored temporarily in session state.")