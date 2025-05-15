import os
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

# Import the database models from the original database.py
from utils.database import Base, Prediction, Patient

@st.cache_resource
def init_local_db(db_path="diretina.db", _show_messages=False):
    """
    Initialize a local SQLite database connection.
    
    Args:
        db_path: Path to SQLite database file (default: diretina.db in current directory)
        _show_messages: If True, show status messages
    
    Returns:
        A dictionary with SQLAlchemy session and engine
    """
    try:
        # Create SQLite database URL
        database_url = f"sqlite:///{db_path}"
        
        if _show_messages:
            st.info(f"Using local SQLite database at: {db_path}")
            
        # Create engine
        engine = create_engine(database_url, echo=False)
        Session = sessionmaker(bind=engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        
        if _show_messages:
            st.success("Successfully connected to local SQLite database")
            
        return {
            'engine': engine,
            'Session': Session
        }
    except Exception as e:
        # Only show error messages if _show_messages is True
        if _show_messages:
            st.error(f"Error connecting to local database: {str(e)}")
            
        # Fallback to in-memory SQLite if file access fails
        if _show_messages:
            st.warning("Falling back to in-memory SQLite database. Your data will not be persisted.")
            
        engine = create_engine("sqlite:///memory:")
        Session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        
        return {
            'engine': engine,
            'Session': Session
        }

# Function to add a prediction to the local database
def add_local_prediction(image_name, prediction, confidence, notes=None, db_path="diretina.db", _show_messages=False):
    """
    Add a prediction record to the local database
    
    Args:
        image_name: Name of the image file
        prediction: 0 for normal, 1 for myopia
        confidence: Confidence score (0-1)
        notes: Optional notes about the prediction
        db_path: Path to SQLite database file
        _show_messages: If True, show status messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = init_local_db(db_path=db_path, _show_messages=_show_messages)
        session = db['Session']()
        
        new_prediction = Prediction(
            image_name=image_name,
            prediction=prediction,
            confidence=confidence,
            notes=notes
        )
        
        session.add(new_prediction)
        session.commit()
        session.close()
        return True
    except Exception as e:
        if _show_messages:
            st.error(f"Error adding prediction to local database: {str(e)}")
        return False

# Function to get all predictions from the local database
def get_all_local_predictions(db_path="diretina.db", _show_messages=False):
    """
    Get all predictions from the local database
    
    Args:
        db_path: Path to SQLite database file
        _show_messages: If True, show status messages
    
    Returns:
        Pandas DataFrame with all predictions
    """
    try:
        db = init_local_db(db_path=db_path, _show_messages=_show_messages)
        session = db['Session']()
        
        predictions = session.query(Prediction).all()
        prediction_dicts = [p.to_dict() for p in predictions]
        
        session.close()
        
        if prediction_dicts:
            return pd.DataFrame(prediction_dicts)
        else:
            # Use dictionary to create empty DataFrame with proper columns
            return pd.DataFrame({
                'id': [], 'timestamp': [], 'image_name': [], 
                'prediction': [], 'confidence': [], 'notes': []
            })
    except Exception as e:
        if _show_messages:
            st.error(f"Error retrieving predictions from local database: {str(e)}")
        # Return empty DataFrame with proper columns
        return pd.DataFrame({
            'id': [], 'timestamp': [], 'image_name': [], 
            'prediction': [], 'confidence': [], 'notes': []
        })

# Function to add a patient to the local database
def add_local_patient(patient_id, name=None, age=None, gender=None, db_path="diretina.db", _show_messages=False):
    """
    Add a patient record to the local database
    
    Args:
        patient_id: Unique patient identifier
        name: Patient name (optional)
        age: Patient age (optional)
        gender: Patient gender (optional)
        db_path: Path to SQLite database file
        _show_messages: If True, show status messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = init_local_db(db_path=db_path, _show_messages=_show_messages)
        session = db['Session']()
        
        new_patient = Patient(
            patient_id=patient_id,
            name=name,
            age=age,
            gender=gender
        )
        
        session.add(new_patient)
        session.commit()
        session.close()
        return True
    except Exception as e:
        if _show_messages:
            st.error(f"Error adding patient to local database: {str(e)}")
        return False

# Function to get all patients from the local database
def get_all_local_patients(db_path="diretina.db", _show_messages=False):
    """
    Get all patients from the local database
    
    Args:
        db_path: Path to SQLite database file
        _show_messages: If True, show status messages
        
    Returns:
        Pandas DataFrame with all patients
    """
    try:
        db = init_local_db(db_path=db_path, _show_messages=_show_messages)
        session = db['Session']()
        
        patients = session.query(Patient).all()
        patient_dicts = [p.to_dict() for p in patients]
        
        session.close()
        
        if patient_dicts:
            return pd.DataFrame(patient_dicts)
        else:
            # Use dictionary to create empty DataFrame with proper columns
            return pd.DataFrame({
                'id': [], 'patient_id': [], 'name': [], 
                'age': [], 'gender': [], 'created_at': []
            })
    except Exception as e:
        if _show_messages:
            st.error(f"Error retrieving patients from local database: {str(e)}")
        # Return empty DataFrame with proper columns
        return pd.DataFrame({
            'id': [], 'patient_id': [], 'name': [], 
            'age': [], 'gender': [], 'created_at': []
        })

# Function to check local database connection
def check_local_db_connection(db_path="diretina.db", _show_messages=False):
    """
    Check if the local database connection is working
    
    Args:
        db_path: Path to SQLite database file
        _show_messages: If True, show status messages
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        db = init_local_db(db_path=db_path, _show_messages=_show_messages)
        engine = db['engine']
        
        # Try a simple query to verify connection
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1"))
            result.fetchone()  # Actually fetch the data to confirm connection works
            
        if _show_messages:
            st.success("Local database connection successful!")
            
        return True
    except Exception as e:
        if _show_messages:
            st.error(f"Error connecting to local database: {str(e)}")
        return False