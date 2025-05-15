import os
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

# Create a base class for SQLAlchemy models
Base = declarative_base()

# Define the Prediction model for storing prediction history
class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    image_name = Column(String, nullable=False)
    prediction = Column(Integer)  # 0 for normal, 1 for myopia
    confidence = Column(Float)
    notes = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'image_name': self.image_name,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'notes': self.notes
        }
        
# Define the Patient model for storing patient information
class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# Initialize database connection
@st.cache_resource
def init_db():
    """
    Initialize database connection and create tables if they don't exist.
    Uses DATABASE_URL environment variable for the connection.
    
    Returns:
        A dictionary with SQLAlchemy session and engine
    """
    try:
        # Get database URL from environment or use a default SQLite URL for development
        database_url = os.environ.get('DATABASE_URL')
        
        if database_url is None:
            st.warning("No database URL found. Please set the DATABASE_URL environment variable.")
            st.info("Using in-memory SQLite database for now. Your data will not be persisted.")
            # Use SQLite for development when no DATABASE_URL is provided
            database_url = "sqlite:///memory:"
        
        # Create engine and session
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        
        return {
            'engine': engine,
            'Session': Session
        }
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        # Fallback to in-memory SQLite
        engine = create_engine("sqlite:///memory:")
        Session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        return {
            'engine': engine,
            'Session': Session
        }

# Function to add a prediction to the database
def add_prediction(image_name, prediction, confidence, notes=None):
    """
    Add a prediction record to the database
    
    Args:
        image_name: Name of the image file
        prediction: 0 for normal, 1 for myopia
        confidence: Confidence score (0-1)
        notes: Optional notes about the prediction
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = init_db()
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
        st.error(f"Error adding prediction to database: {str(e)}")
        return False

# Function to get all predictions from the database
def get_all_predictions():
    """
    Get all predictions from the database
    
    Returns:
        Pandas DataFrame with all predictions
    """
    try:
        db = init_db()
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
        st.error(f"Error retrieving predictions from database: {str(e)}")
        # Use dictionary to create empty DataFrame with proper columns
        return pd.DataFrame({
            'id': [], 'timestamp': [], 'image_name': [], 
            'prediction': [], 'confidence': [], 'notes': []
        })

# Function to add a patient to the database
def add_patient(patient_id, name=None, age=None, gender=None):
    """
    Add a patient record to the database
    
    Args:
        patient_id: Unique patient identifier
        name: Patient name (optional)
        age: Patient age (optional)
        gender: Patient gender (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = init_db()
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
        st.error(f"Error adding patient to database: {str(e)}")
        return False

# Function to get all patients from the database
def get_all_patients():
    """
    Get all patients from the database
    
    Returns:
        Pandas DataFrame with all patients
    """
    try:
        db = init_db()
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
        st.error(f"Error retrieving patients from database: {str(e)}")
        # Use dictionary to create empty DataFrame with proper columns
        return pd.DataFrame({
            'id': [], 'patient_id': [], 'name': [], 
            'age': [], 'gender': [], 'created_at': []
        })

# Function to check database connection
def check_db_connection():
    """
    Check if the database connection is working
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        db = init_db()
        engine = db['engine']
        
        # Try a simple query
        with engine.connect() as conn:
            conn.execute("SELECT 1")
            
        return True
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return False