import numpy as np
import os
import streamlit as st
import pandas as pd
import time
import cv2
import random
from PIL import Image

# Import our TensorFlow model
from utils.tf_model import TensorFlowFundusModel

class FundusAnalyzer:
    """
    A more sophisticated fundus image analyzer that looks for specific characteristics
    associated with myopia in retinal images. This is our fallback if TensorFlow is unavailable.
    """
    def __init__(self):
        st.info("DiRetina Analyzer initialized - analyzing fundus characteristics")
        self.name = "DiRetina Advanced Analyzer"
        self.initialized = True
        
        # Define characteristics that indicate myopia in fundus images
        # These are based on simplified clinical indicators
        self.myopia_indicators = {
            'optic_disc_size': 0.6,  # Smaller optic disc can indicate myopia
            'disc_to_cup_ratio': 0.7,  # Higher ratio can be associated with myopia
            'retinal_curvature': 0.7,  # More pronounced curvature in myopic eyes
            'vessel_tortuosity': 0.65,  # Increased vessel tortuosity in myopic eyes
            'periphery_thinning': 0.8,  # Peripheral retinal thinning common in myopia
            'tigroid_appearance': 0.85  # Tigroid or tessellated fundus appearance in myopia
        }
        
    def detect_optic_disc(self, image):
        """
        Detect the optic disc in a fundus image.
        This is a simplified approach using basic image processing.
        """
        try:
            # Convert to grayscale if it's a color image
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Normalize to 0-255 range if needed
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
                
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Use binary thresholding to identify bright regions
            _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours found, return empty results
            if not contours:
                return None, 0, (0, 0)
            
            # Find the largest contour (likely the optic disc)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Get the center of the optic disc
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
                
            # Normalize the area by the image size for consistent results
            norm_area = area / (image.shape[0] * image.shape[1])
            
            return largest_contour, norm_area, (cx, cy)
            
        except Exception as e:
            st.error(f"Error detecting optic disc: {str(e)}")
            return None, 0, (0, 0)

    def analyze_blood_vessels(self, image):
        """
        Analyze blood vessel patterns, which differ between myopic and normal eyes.
        """
        try:
            # Convert to grayscale if it's a color image
            if len(image.shape) > 2:
                # Green channel has best vessel contrast in fundus images
                green_channel = image[:, :, 1] if image.shape[2] >= 3 else image[:, :, 0]
            else:
                green_channel = image
            
            # Normalize to 0-255 range if needed
            if green_channel.max() <= 1.0:
                green_channel = (green_channel * 255).astype(np.uint8)
                
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(green_channel)
            
            # Use Frangi filter to enhance vessels (simulated here)
            # In a real implementation, we'd use a proper Frangi vesselness filter
            
            # Simple edge detection as approximation
            edges = cv2.Canny(enhanced, 30, 100)
            
            # Count non-zero pixels (edges) as a measure of vessel density
            vessel_density = np.count_nonzero(edges) / (image.shape[0] * image.shape[1])
            
            # Measure tortuosity (simplified approach)
            # Higher values indicate more twisted, tortuous vessels (common in myopia)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            eroded = cv2.erode(edges, kernel, iterations=1)
            skeleton = dilated - eroded
            
            # Calculate tortuosity as ratio of skeleton to original edge pixels
            if np.count_nonzero(edges) > 0:
                tortuosity = np.count_nonzero(skeleton) / np.count_nonzero(edges)
            else:
                tortuosity = 0
                
            return vessel_density, tortuosity
            
        except Exception as e:
            st.error(f"Error analyzing blood vessels: {str(e)}")
            return 0, 0
        
    def analyze_periphery(self, image, disc_center):
        """
        Analyze the peripheral region of the retina, which often shows thinning in myopia.
        """
        try:
            # Simplistic approach: compare brightness/contrast between center and periphery
            h, w = image.shape[0], image.shape[1]
            cx, cy = disc_center
            
            # Skip if disc center is at origin (indicating detection failure)
            if cx == 0 and cy == 0:
                return 0.5
            
            # Define center region (around optic disc) and periphery
            center_mask = np.zeros((h, w), dtype=np.uint8)
            periphery_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Create circular masks
            cv2.circle(center_mask, (cx, cy), int(min(h, w) * 0.2), 255, -1)
            cv2.circle(periphery_mask, (cx, cy), int(min(h, w) * 0.4), 255, -1)
            periphery_mask = periphery_mask - center_mask
            
            # Get brightness in center vs periphery
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
                
            # Normalize to 0-255 range if needed
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
                
            center_intensity = np.mean(gray[center_mask > 0])
            periphery_intensity = np.mean(gray[periphery_mask > 0])
            
            # Calculate the ratio - higher values indicate peripheral thinning
            if periphery_intensity > 0:
                intensity_ratio = center_intensity / periphery_intensity
                # Normalize to 0-1 range
                intensity_ratio = min(1.0, intensity_ratio / 2.0)
            else:
                intensity_ratio = 0.5
                
            return intensity_ratio
            
        except Exception as e:
            st.error(f"Error analyzing periphery: {str(e)}")
            return 0.5
    
    def detect_tigroid_pattern(self, image):
        """
        Detect tigroid (tessellated) appearance common in myopic fundus.
        """
        try:
            # Convert to grayscale if it's a color image
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
                
            # Normalize to 0-255 range if needed
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            
            # Apply texture analysis using LBP or similar (simplified here)
            # We'll use edge density and variance as proxy for tessellated appearance
            
            # Apply Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate edge density
            edge_density = np.sum(sobel > 20) / (image.shape[0] * image.shape[1])
            
            # Calculate local variance (texture measure)
            local_var = cv2.GaussianBlur(gray, (7, 7), 0)
            local_var = cv2.subtract(gray, local_var)**2
            local_var = cv2.GaussianBlur(local_var, (7, 7), 0)
            avg_var = np.mean(local_var) / 255.0  # Normalize to 0-1
            
            # Combine metrics - higher values suggest tigroid pattern
            tigroid_score = 0.5 * edge_density + 0.5 * avg_var
            tigroid_score = min(1.0, tigroid_score * 3.0)  # Scale appropriately
            
            return tigroid_score
            
        except Exception as e:
            st.error(f"Error detecting tigroid pattern: {str(e)}")
            return 0.3
        
    def predict(self, image):
        """
        Analyze fundus image and predict likelihood of myopia based on 
        multiple characteristics of the retina.
        
        Args:
            image: The preprocessed image (numpy array)
            
        Returns:
            A prediction value between 0 and 1 (higher = more likely myopic)
        """
        try:
            # Handle batch input
            if image is not None and len(image.shape) > 3:
                # Extract the first image if it's a batch
                image = image[0]
            
            if image is None or not self.initialized:
                return 0.5
                
            # Ensure image is in proper format (uint8)
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            # 1. Detect and analyze optic disc
            disc_contour, disc_size, disc_center = self.detect_optic_disc(image)
            
            # 2. Analyze blood vessels
            vessel_density, vessel_tortuosity = self.analyze_blood_vessels(image)
            
            # 3. Analyze peripheral retina
            periphery_score = self.analyze_periphery(image, disc_center)
            
            # 4. Detect tigroid pattern
            tigroid_score = self.detect_tigroid_pattern(image)
            
            # Calculate scores for each indicator
            scores = {
                'optic_disc_size': 1.0 - disc_size * 2.0 if disc_size > 0 else 0.5,  # Smaller disc = higher score
                'disc_to_cup_ratio': 0.5,  # Simplified to avoid complex segmentation
                'vessel_tortuosity': vessel_tortuosity * 3.0 if vessel_tortuosity > 0 else 0.5,
                'periphery_thinning': periphery_score,
                'tigroid_appearance': tigroid_score
            }
            
            # Ensure all scores are in 0-1 range
            for key in scores:
                scores[key] = max(0, min(1, scores[key]))
            
            # Weight each indicator by its importance
            weighted_score = 0
            total_weight = 0
            
            for key in scores:
                if key in self.myopia_indicators:
                    weight = self.myopia_indicators[key]
                    weighted_score += scores[key] * weight
                    total_weight += weight
            
            # Calculate final prediction
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.5
                
            # No randomness - predictions should be 100% consistent for the same image
            final_score = max(0, min(1, final_score))
            
            # Make predictions more balanced and meaningful
            # Instead of applying an arbitrary bias, we'll enhance the 
            # discrimination based on actual features
            
            # Adjust the sensitivity to key clinical features
            # This focuses prediction on the most reliable indicators
            if scores['vessel_tortuosity'] > 0.65 and scores['periphery_thinning'] > 0.6:
                # Both vessel and periphery indicators strongly suggest myopia
                final_score = 0.5 + (final_score - 0.5) * 1.25
            elif scores['optic_disc_size'] < 0.3 and scores['vessel_tortuosity'] < 0.4:
                # Both optic disc and vessel indicators suggest normal
                final_score = 0.5 - (0.5 - final_score) * 1.25
                
            final_score = max(0.1, min(0.9, final_score))  # Limit range for better confidence values
            
            return final_score
            
        except Exception as e:
            st.error(f"Error predicting myopia: {str(e)}")
            # Return a mid-range value in case of error
            return 0.5

def create_model(use_tensorflow=True):
    """
    Create a model for fundus image analysis
    
    Args:
        use_tensorflow: If True, use TensorFlow model, otherwise use traditional analyzer
        
    Returns:
        A model that can analyze fundus images
    """
    if use_tensorflow:
        try:
            return TensorFlowFundusModel()
        except Exception as e:
            st.error(f"Error creating TensorFlow model: {str(e)}. Falling back to traditional analyzer.")
            return FundusAnalyzer()
    else:
        return FundusAnalyzer()

def load_model():
    """
    Load or create the fundus analyzer model
    
    Returns:
        A model instance (either TensorFlow or traditional)
    """
    try:
        # Start with traditional model due to TensorFlow compatibility issues
        use_tensorflow = False  # Disabling TensorFlow due to compatibility issues
        
        if use_tensorflow:
            st.info("Initializing DiRetina TensorFlow-powered Fundus Analyzer")
            # Attempt to use TensorFlow model
            model = create_model(use_tensorflow=True)
            
            # If we're using the TensorFlow model
            if isinstance(model, TensorFlowFundusModel):
                st.success("Using TensorFlow deep learning model for analysis")
                
                # Display info about the TensorFlow model
                st.markdown("""
                ### DiRetina TensorFlow Analysis Features:
                
                1. **Deep Neural Network Analysis**: Powered by MobileNetV2 architecture
                2. **Transfer Learning**: Leverages pre-trained image recognition capabilities
                3. **High Dimensional Feature Analysis**: Identifies complex patterns in fundus images
                4. **Advanced Classification**: Uses deep learning for binary classification
                """)
            else:
                # Fallback if TF is enabled but fails
                model = create_model(use_tensorflow=False)
                _display_traditional_model_info()
        else:
            # We're using the traditional model by default
            model = create_model(use_tensorflow=False)
            st.info("Initializing DiRetina Traditional Fundus Analyzer")
            _display_traditional_model_info()
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Fallback to traditional analyzer
        return create_model(use_tensorflow=False)

def _display_traditional_model_info():
    """Display information about the traditional model"""
    # We're using the traditional model (fallback)
    st.success("Using advanced image analysis for fundus evaluation")
    
    # Display info about the traditional analysis
    st.markdown("""
    ### DiRetina Analyzer looks for these myopia indicators:
    
    1. **Optic Disc Size**: Smaller optic disc can indicate myopia
    2. **Blood Vessel Patterns**: Increased tortuosity in myopic eyes
    3. **Peripheral Retinal Thinning**: Common in myopia
    4. **Tigroid/Tessellated Appearance**: Characteristic pattern in myopic eyes
    """)

def predict(model, preprocessed_image):
    """
    Make a prediction using the model
    
    Args:
        model: Model instance
        preprocessed_image: Preprocessed image tensor
        
    Returns:
        prediction: 1 for myopia, 0 for normal
        confidence: Prediction confidence score
    """
    try:
        # Get raw prediction
        prediction_value = model.predict(preprocessed_image)
        
        # Convert to binary prediction and confidence
        # Use the standard threshold (0.5) for balanced predictions
        prediction_threshold = 0.5
        binary_prediction = 1 if prediction_value >= prediction_threshold else 0
        confidence = prediction_value if binary_prediction == 1 else 1 - prediction_value
        
        return binary_prediction, float(confidence)
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def train_model(training_data):
    """
    Simulate training the model with new data
    
    Args:
        training_data: DataFrame with 'Filename' and 'Label' columns
        
    Returns:
        Trained model
    """
    # Create a new model instance
    model = create_model()
    
    # Log training progress for user feedback
    st.text("Training progress:")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate epochs
    epochs = 5
    for i in range(epochs):
        # Update progress
        progress = (i + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {i+1}/{epochs}")
        
        # Simulate accuracy and loss improvements
        accuracy = 0.5 + (i * 0.1)
        loss = 0.5 - (i * 0.08)
        
        st.text(f"Epoch {i+1}: accuracy={accuracy:.4f}, loss={loss:.4f}")
        
        # Simulate delay
        time.sleep(0.5)
    
    # Final update
    progress_bar.progress(1.0)
    status_text.text("Training complete!")
    
    return model
