import streamlit as st
from datetime import datetime
import io
from PIL import Image
import numpy as np
from utils.image_processing import preprocess_image, validate_fundus_image
from utils.model import predict
from utils.database import add_prediction

def create_upload_section():
    """
    Create the image upload and prediction section
    """
    st.title("Upload & Predict")
    st.markdown("### Upload fundus images for myopia detection")
    
    # Initialize session state for tracking processed files
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    # Image upload widget
    uploaded_files = st.file_uploader(
        "Upload one or more fundus images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # Process uploaded images if any
    if uploaded_files:
        # Create a list of newly uploaded files that haven't been processed yet
        new_files = []
        for uploaded_file in uploaded_files:
            # Generate a unique identifier for this file (name + size + last modified)
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Only process files we haven't seen before
            if file_id not in st.session_state.processed_files:
                new_files.append((uploaded_file, file_id))
                # Mark this file as processed
                st.session_state.processed_files.add(file_id)
        
        # Show message if all files have been processed already
        if not new_files and uploaded_files:
            st.info("All uploaded images have already been processed. Upload new images to analyze them.")
        
        # Process only new files
        for uploaded_file, file_id in new_files:
            # Read file
            image_bytes = uploaded_file.getvalue()
            
            # Validate if it's a fundus image
            is_valid = validate_fundus_image(image_bytes)
            
            if not is_valid:
                st.warning(f"The uploaded image '{uploaded_file.name}' doesn't appear to be a valid fundus image. Processing anyway.")
            
            # Display the image
            st.image(image_bytes, caption=f"Uploaded: {uploaded_file.name}", width=300)
            
            # Process image and make prediction
            with st.spinner(f"Analyzing image: {uploaded_file.name}..."):
                # Preprocess image
                preprocessed_image = preprocess_image(image_bytes)
                
                if preprocessed_image is not None:
                    # Make prediction
                    prediction, confidence = predict(st.session_state.model, preprocessed_image)
                    
                    if prediction is not None:
                        # Store prediction results in session state
                        st.session_state.predictions.append((prediction, confidence))
                        
                        # Create timestamp
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Add to history with timestamp
                        st.session_state.prediction_history.append({
                            'timestamp': timestamp,
                            'image_name': uploaded_file.name,
                            'prediction': prediction,
                            'confidence': confidence
                        })
                        
                        # Update last upload time
                        st.session_state.last_upload_time = datetime.now()
                        
                        # Save to database - don't show error messages here
                        db_success = add_prediction(
                            image_name=uploaded_file.name,
                            prediction=prediction,
                            confidence=confidence,
                            _show_messages=False
                        )
                        
                        # We still have the prediction in session state even if database connection fails
                        
                        # Display result
                        if prediction == 1:
                            # Ensure confidence is a number
                            conf_value = float(confidence) if confidence is not None else 0.0
                            st.error(f"⚠️ Myopia detected with {conf_value*100:.1f}% confidence")
                            
                            # Additional information for myopia
                            with st.expander("What is Myopia?"):
                                st.markdown("""
                                **Myopia (Nearsightedness)** is a common vision condition where you can see objects near to you clearly, but objects farther away are blurry.
                                
                                **Possible symptoms:**
                                - Blurry vision when looking at distant objects
                                - Squinting to see clearly
                                - Eye strain and headaches
                                
                                **Recommendations:**
                                - Further clinical evaluation is recommended
                                - Regular eye examinations
                                - Corrective lenses (glasses or contact lenses)
                                - In some cases, refractive surgery
                                """)
                        else:
                            # Ensure confidence is a number
                            conf_value = float(confidence) if confidence is not None else 0.0
                            st.success(f"✅ Normal eye detected with {conf_value*100:.1f}% confidence")
                    else:
                        st.error(f"Error making prediction for {uploaded_file.name}. Please try again.")
                else:
                    st.error(f"Error preprocessing image {uploaded_file.name}. Please try a different image.")
    
    # Instructions and examples
    else:
        st.info(
            "Upload fundus images to detect myopia using our AI model. "
            "The model analyzes the retinal features to identify signs of myopia."
        )
        

    
    # Show tips for good fundus images
    with st.expander("Tips for Good Fundus Images"):
        st.markdown("""
        For the best analysis results, ensure your fundus images:
        
        1. **Are clear and in focus** - blurry images reduce accuracy
        2. **Have good lighting** - evenly illuminated with good contrast
        3. **Are properly centered** - with the optic disc visible
        4. **Have minimal artifacts** - like dust or scratches
        5. **Are in color** - color images provide more diagnostic information
        """)
