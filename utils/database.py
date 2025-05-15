import os
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import urllib.parse
import re

def fix_supabase_url(url):
    """
    Fix common issues with Supabase connection strings.
    
    Args:
        url: The original database URL
        
    Returns:
        Fixed URL string
    """
    # First, detect if this is a pooler URL with the specific format we're seeing
    if 'pooler.supabase' in url and '@' in url:
        # This looks like: 
        # postgresql://postgres.urahevcropdwuyqqnpny:[@Weareyoung256]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
        
        # Split by @ symbols - we expect 2 of them
        parts = url.split('@')
        
        if len(parts) >= 3:
            # We have more than one @ symbol, so we need to carefully reconstruct
            
            # Extract protocol and username (before first colon)
            protocol_user_part = parts[0]
            if ':' in protocol_user_part:
                protocol_user = protocol_user_part.split(':')[0]
            else:
                protocol_user = protocol_user_part
                
            # Extract password - everything between first : and second @, removing brackets
            password_part = url.split(':', 1)[1].split('@', 1)[0]
            password = password_part.replace('[', '').replace(']', '')
            
            # Extract host and the rest - everything after the last @
            host_part = parts[-1]
            
            # Reconstruct with proper URL encoding for the password
            encoded_password = urllib.parse.quote_plus(password)
            
            # Make sure we're using the pooler hostname and correct port
            if 'pooler.supabase' not in host_part and ':6543' not in host_part:
                # This might be using the direct hostname instead of pooler
                st.warning("Detected non-pooler hostname, attempting to fix")
                
                # Try to extract the project ID from the hostname
                if '.' in host_part:
                    parts = host_part.split('.')
                    project_id = None
                    for part in parts:
                        if len(part) > 10 and not part.startswith('supabase'):
                            project_id = part
                            break
                    
                    if project_id:
                        # Replace with proper pooler hostname
                        host_part = f"aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
                        st.info(f"Using pooler hostname: {host_part}")
            
            # Ensure we're using pgbouncer=true for pooler connections
            if 'pooler.supabase' in host_part and '?' not in host_part:
                host_part = host_part + "?pgbouncer=true"
                st.info("Added pgbouncer=true parameter")
                
            fixed_url = f"{protocol_user}:{encoded_password}@{host_part}"
            
            # Add postgresql:// if it was stripped
            if not fixed_url.startswith('postgresql://'):
                fixed_url = 'postgresql://' + fixed_url
                
            return fixed_url
            
    # Default approach for simpler URLs
    if '[' in url and ']' in url:
        url = url.replace('[', '').replace(']', '')
    
    # Handle URLs with a more standard format
    parts = re.split(r'(?<!\\)@', url)  # Split by @ but not \@
    
    if len(parts) == 2:
        # Standard URL with one @ symbol
        auth, rest = parts
        if ':' in auth:
            user, pwd = auth.rsplit(':', 1)
            pwd = urllib.parse.quote_plus(pwd)
            return f"{user}:{pwd}@{rest}"
    
    # If we couldn't parse it specially, at least encode any special chars
    try:
        parsed = urllib.parse.urlparse(url)
        userinfo = parsed.netloc.split('@', 1)[0]
        if ':' in userinfo:
            username, password = userinfo.split(':', 1)
            encoded_password = urllib.parse.quote_plus(password)
            new_netloc = f"{username}:{encoded_password}@{parsed.netloc.split('@', 1)[1]}"
            parsed = parsed._replace(netloc=new_netloc)
            return urllib.parse.urlunparse(parsed)
    except:
        pass
        
    # If all special handling fails, return the original
    return url

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
        
        # For Supabase PostgreSQL connections, we need to fix the URL format
        if 'supabase' in database_url.lower() or 'pooler.supabase' in database_url.lower():
            st.info("Detected Supabase PostgreSQL connection")
            
            try:
                # Parse and fix the Supabase URL with a specialized function
                database_url = fix_supabase_url(database_url)
                st.success("Successfully formatted Supabase connection string")
            except Exception as e:
                st.error(f"Error formatting database URL: {str(e)}")
                st.info("You might need to manually format your DATABASE_URL properly")
            
        # Create engine with proper dialect options for PostgreSQL
        if 'postgresql' in database_url.lower():
            # Connect with minimal options for maximum compatibility
            engine = create_engine(
                database_url,
                # Echo SQL for debugging during development
                echo=True
            )
            st.info(f"Connected to PostgreSQL database")
        else:
            # For other database types (e.g., SQLite)
            engine = create_engine(database_url)
            st.info(f"Using SQLite database")
            
        Session = sessionmaker(bind=engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
        
        return {
            'engine': engine,
            'Session': Session
        }
    except Exception as e:
        # Get more detailed error information
        import traceback
        error_details = traceback.format_exc()
        
        # Display error with troubleshooting information
        st.error(f"Error connecting to database: {str(e)}")
        st.error("Please check your DATABASE_URL format")
        
        # Show some helpful tips
        st.markdown("""
        ### Database Connection Troubleshooting:
        
        1. For **Supabase**, use the **Transaction Pooler** connection string (URI format)
        2. Make sure you've replaced `[YOUR-PASSWORD]` with your actual database password
        3. Check if your Supabase project is active and not in pause state
        4. Verify that your IP is allowed in Supabase network restrictions
        
        **Sample format for Supabase:**
        ```
        postgresql://postgres:your_password@db.example.supabase.co:6543/postgres?pgbouncer=true
        ```
        
        **Detailed error:**
        """)
        # Show detailed error in an expander (helps with troubleshooting without cluttering the UI)
        with st.expander("Show detailed error"):
            st.code(error_details)
            
        # Fallback to in-memory SQLite
        engine = create_engine("sqlite:///memory:")
        Session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        
        # Return the fallback engine and session
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
        
        # Try a simple query to verify connection
        with engine.connect() as conn:
            # SQLAlchemy 2.0 requires using text() for raw SQL
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1"))
            result.fetchone()  # Actually fetch the data to confirm connection works
            
        # Get database type for information
        db_type = "PostgreSQL" if 'postgresql' in str(engine.url).lower() else "SQLite"
        
        # If it's SQLite in-memory, we're using the fallback
        if db_type == "SQLite" and ':memory:' in str(engine.url) or 'sqlite:///memory:' in str(engine.url):
            # We're using the fallback SQLite database
            return False
            
        # Connection successful
        return True
    except Exception as e:
        # Log the error for troubleshooting
        import traceback
        print(f"Database connection error: {str(e)}")
        print(traceback.format_exc())
        return False