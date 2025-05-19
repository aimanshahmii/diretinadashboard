import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Import database models from the main database module
from utils.database import Base, Prediction, Patient

def init_mysql_db(mysql_url=None, _show_messages=False):
    """
    Initialize MySQL database connection and create tables if they don't exist.
    
    Args:
        mysql_url: MySQL connection URL in format mysql://username:password@host:port/database
                  If None, will try to use environment variable MYSQL_DATABASE_URL
        _show_messages: If True, show status messages
    
    Returns:
        A dictionary with SQLAlchemy session and engine
    """
    if mysql_url is None:
        # Try to get from environment or session state
        mysql_url = os.getenv("MYSQL_DATABASE_URL")
        
        if mysql_url is None and "mysql_url" in st.session_state:
            mysql_url = st.session_state.mysql_url
            
    if mysql_url is None:
        if _show_messages:
            st.error("MySQL connection URL not provided. Please configure database connection.")
        return None
    
    try:
        # Store for later use
        st.session_state.mysql_url = mysql_url
        
        # Create database engine
        engine = create_engine(mysql_url)
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        
        # Create session factory
        Session = sessionmaker(bind=engine)
        session = Session()
        
        if _show_messages:
            st.success("Connected to MySQL database successfully!")
            
        return {"engine": engine, "session": session}
    
    except Exception as e:
        if _show_messages:
            st.error(f"Failed to connect to MySQL database: {str(e)}")
        return None

def add_mysql_prediction(image_name, prediction, confidence, notes=None, _show_messages=False):
    """
    Add a prediction record to the MySQL database
    
    Args:
        image_name: Name of the image file
        prediction: 0 for normal, 1 for myopia
        confidence: Confidence score (0-1)
        notes: Optional notes about the prediction
        _show_messages: If True, show status messages
        
    Returns:
        True if successful, False otherwise
    """
    db = init_mysql_db(_show_messages=_show_messages)
    if db is None:
        return False
    
    try:
        # Create a new Prediction record
        new_prediction = Prediction(
            image_name=image_name,
            prediction=prediction,
            confidence=confidence,
            notes=notes
        )
        
        # Add to database
        db["session"].add(new_prediction)
        db["session"].commit()
        
        if _show_messages:
            st.success(f"Added prediction for {image_name} to database")
            
        return True
    
    except Exception as e:
        if _show_messages:
            st.error(f"Failed to add prediction to database: {str(e)}")
        return False
    
    finally:
        db["session"].close()

def get_all_mysql_predictions(_show_messages=False):
    """
    Get all predictions from the MySQL database
    
    Args:
        _show_messages: If True, show status messages
    
    Returns:
        Pandas DataFrame with all predictions
    """
    db = init_mysql_db(_show_messages=_show_messages)
    if db is None:
        return pd.DataFrame()
    
    try:
        # Query all predictions
        predictions = db["session"].query(Prediction).all()
        
        # Convert to list of dictionaries
        prediction_data = [p.to_dict() for p in predictions]
        
        # Convert to DataFrame
        df = pd.DataFrame(prediction_data)
        
        if len(df) > 0:
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Add a Time column for easy display
            df['Time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add Result text column
            df['Result'] = df['prediction'].apply(lambda x: 'Myopia' if x == 1 else 'Normal')
            
        return df
    
    except Exception as e:
        if _show_messages:
            st.error(f"Failed to get predictions from database: {str(e)}")
        return pd.DataFrame()
    
    finally:
        db["session"].close()

def add_mysql_patient(patient_id, name=None, age=None, gender=None, _show_messages=False):
    """
    Add a patient record to the MySQL database
    
    Args:
        patient_id: Unique patient identifier
        name: Patient name (optional)
        age: Patient age (optional)
        gender: Patient gender (optional)
        _show_messages: If True, show status messages
        
    Returns:
        True if successful, False otherwise
    """
    db = init_mysql_db(_show_messages=_show_messages)
    if db is None:
        return False
    
    try:
        # Create a new Patient record
        new_patient = Patient(
            patient_id=patient_id,
            name=name,
            age=age,
            gender=gender
        )
        
        # Add to database
        db["session"].add(new_patient)
        db["session"].commit()
        
        if _show_messages:
            st.success(f"Added patient {patient_id} to database")
            
        return True
    
    except Exception as e:
        if _show_messages:
            st.error(f"Failed to add patient to database: {str(e)}")
        return False
    
    finally:
        db["session"].close()

def get_all_mysql_patients(_show_messages=False):
    """
    Get all patients from the MySQL database
    
    Args:
        _show_messages: If True, show status messages
    
    Returns:
        Pandas DataFrame with all patients
    """
    db = init_mysql_db(_show_messages=_show_messages)
    if db is None:
        return pd.DataFrame()
    
    try:
        # Query all patients
        patients = db["session"].query(Patient).all()
        
        # Convert to list of dictionaries
        patient_data = [p.to_dict() for p in patients]
        
        # Convert to DataFrame
        df = pd.DataFrame(patient_data)
        
        if len(df) > 0:
            # Convert created_at to datetime if it's not already
            if 'created_at' in df.columns and not pd.api.types.is_datetime64_dtype(df['created_at']):
                df['created_at'] = pd.to_datetime(df['created_at'])
                
            # Add a Registered Date column for easy display
            df['Registered Date'] = df['created_at'].dt.strftime('%Y-%m-%d')
            
        return df
    
    except Exception as e:
        if _show_messages:
            st.error(f"Failed to get patients from database: {str(e)}")
        return pd.DataFrame()
    
    finally:
        db["session"].close()

def check_mysql_db_connection(_show_messages=False):
    """
    Check if the MySQL database connection is working
    
    Args:
        _show_messages: If True, show status messages
    
    Returns:
        True if connection is successful, False otherwise
    """
    db = init_mysql_db(_show_messages=_show_messages)
    if db is None:
        return False
    
    try:
        # Try to query something simple
        db["session"].execute("SELECT 1")
        
        if _show_messages:
            st.success("MySQL database connection is working")
            
        return True
    
    except Exception as e:
        if _show_messages:
            st.error(f"MySQL database connection failed: {str(e)}")
        return False
    
    finally:
        db["session"].close()