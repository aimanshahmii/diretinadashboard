import numpy as np
import cv2
from PIL import Image
import io
import streamlit as st

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess a fundus image for the model.
    
    Args:
        image_bytes: Bytes of the uploaded image
        target_size: Target size for model input (default: 224x224)
        
    Returns:
        Preprocessed image ready for model prediction
    """
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        # Convert to RGB if RGBA
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # Resize
        img_array = cv2.resize(img_array, target_size)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        # Expand dimensions to create batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def validate_fundus_image(image_bytes):
    """
    Validate if the uploaded image is likely a fundus image.
    
    Args:
        image_bytes: Bytes of the uploaded image
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if image is in color
        if len(img_array.shape) < 3:
            return False
        
        # Convert to RGB if RGBA
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Check if image has reasonable dimensions
        if img_array.shape[0] < 100 or img_array.shape[1] < 100:
            return False
        
        # Check if image has a circular shape (common for fundus images)
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Check if the largest contour is somewhat circular
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # If the area is too small, it's likely not a fundus image
        if area < 1000:
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Error validating image: {str(e)}")
        return False
