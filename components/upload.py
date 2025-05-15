import streamlit as st
from datetime import datetime
import io
from PIL import Image
import numpy as np
from utils.image_processing import preprocess_image, validate_fundus_image
from utils.model import predict

def create_upload_section():
    """
    Create the image upload and prediction section
    """
    st.title("Upload & Predict")
    st.markdown("### Upload fundus images for myopia detection")
    
    # Image upload widget
    uploaded_files = st.file_uploader(
        "Upload one or more fundus images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # Process uploaded images if any
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read file
            image_bytes = uploaded_file.getvalue()
            
            # Validate if it's a fundus image
            is_valid = validate_fundus_image(image_bytes)
            
            if not is_valid:
                st.warning(f"The uploaded image '{uploaded_file.name}' doesn't appear to be a valid fundus image. Processing anyway.")
            
            # Display the image
            st.image(image_bytes, caption=f"Uploaded: {uploaded_file.name}", width=300)
            
            # Process image and make prediction
            with st.spinner("Analyzing image..."):
                # Preprocess image
                preprocessed_image = preprocess_image(image_bytes)
                
                if preprocessed_image is not None:
                    # Make prediction
                    prediction, confidence = predict(st.session_state.model, preprocessed_image)
                    
                    if prediction is not None:
                        # Store prediction results
                        st.session_state.predictions.append((prediction, confidence))
                        
                        # Add to history with timestamp
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'image_name': uploaded_file.name,
                            'prediction': prediction,
                            'confidence': confidence
                        })
                        
                        # Update last upload time
                        st.session_state.last_upload_time = datetime.now()
                        
                        # Display result
                        if prediction == 1:
                            st.error(f"⚠️ Myopia detected with {confidence*100:.1f}% confidence")
                            
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
                            st.success(f"✅ Normal eye detected with {confidence*100:.1f}% confidence")
                    else:
                        st.error("Error making prediction. Please try again.")
                else:
                    st.error("Error preprocessing image. Please try a different image.")
    
    # Instructions and examples
    else:
        st.info(
            "Upload fundus images to detect myopia using our AI model. "
            "The model analyzes the retinal features to identify signs of myopia."
        )
        
        # Example images
        st.markdown("### Example Fundus Images")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(
                "https://images.unsplash.com/photo-1564367133818-6ac5df7debfc",
                caption="Fundus Example 1",
                width=200
            )
        
        with col2:
            st.image(
                "https://images.unsplash.com/photo-1486649567693-aaa9b2e59385",
                caption="Fundus Example 2",
                width=200
            )
        
        with col3:
            st.image(
                "https://images.unsplash.com/photo-1635012861357-4742bd877c0c",
                caption="Fundus Example 3",
                width=200
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
