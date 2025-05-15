import streamlit as st
import numpy as np
from PIL import Image
from utils.heatmap import generate_heatmap, get_risk_analysis, image_to_base64
from utils.image_processing import preprocess_image
import io

def create_heatmap_view():
    """
    Create the heatmap visualization section for high-risk zones in fundus images
    """
    st.title("High-Risk Zone Analysis")
    st.markdown("### Visualize areas of concern in fundus images")
    
    # Explanation about heatmaps
    with st.expander("About Heatmap Analysis"):
        st.markdown("""
        **What is a heatmap analysis?**
        
        A heatmap visualization highlights areas of potential concern in fundus images. These high-risk zones are identified using image analysis techniques that detect features associated with myopia and other eye conditions:
        
        - **Optic disc changes**: Changes in the shape, size, or appearance of the optic disc
        - **Blood vessel patterns**: Abnormal vessel distribution or tortuosity
        - **Peripheral thinning**: Areas of potential retinal thinning in the periphery
        - **Pigmentary changes**: Unusual pigmentation patterns that may indicate myopic changes
        
        **Color interpretation:**
        - **Red/Yellow areas**: Highest risk regions that require attention
        - **Green/Blue areas**: Lower risk regions or normal features
        
        This tool is intended as a decision support system, and all findings should be verified by a qualified healthcare professional.
        """)
    
    # Upload section for new images
    st.markdown("### Upload a fundus image for heatmap analysis")
    uploaded_file = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"], key="heatmap_uploader")
    
    # Process the uploaded image
    if uploaded_file is not None:
        # Read the image
        image_bytes = uploaded_file.getvalue()
        
        try:
            # Display the original image
            st.image(image_bytes, caption="Uploaded fundus image", width=400)
            
            # Process image for heatmap
            with st.spinner("Analyzing image and generating heatmap..."):
                # Convert image bytes to a numpy array
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                
                # Check if we have a recent prediction confidence
                if (hasattr(st.session_state, 'predictions') and 
                    st.session_state.predictions and 
                    len(st.session_state.predictions) > 0):
                    # Use the most recent prediction confidence
                    _, confidence = st.session_state.predictions[-1]
                else:
                    # Default confidence (moderate)
                    confidence = 0.5
                
                # Generate heatmap
                heatmap_image, risk_areas = generate_heatmap(image_array, confidence)
                
                # Display heatmap
                st.subheader("Heatmap Visualization")
                st.image(heatmap_image, caption="Fundus image with high-risk zone heatmap", use_column_width=True)
                
                # Get risk analysis
                risk_analysis = get_risk_analysis(risk_areas)
                
                # Display risk analysis
                st.subheader("Risk Analysis")
                st.markdown(risk_analysis)
                
                # Display detailed information about what the colors mean
                with st.expander("How to interpret this heatmap"):
                    st.markdown("""
                    ### Heatmap Color Interpretation
                    
                    - **Red zones**: Highest risk areas that may indicate significant myopic changes or other pathology
                    - **Yellow zones**: Moderate risk areas that show potential early signs of abnormality
                    - **Green/Blue zones**: Lower risk or normal areas
                    
                    ### Potential Risk Factors Shown
                    
                    1. **Optic Disc Changes**: 
                       - Enlargement of the optic disc
                       - Peripapillary atrophy (tissue thinning around the disc)
                    
                    2. **Vascular Changes**:
                       - Abnormal vessel patterns
                       - Vessel tortuosity (twisted blood vessels)
                    
                    3. **Peripheral Thinning**:
                       - Areas of reduced retinal thickness
                       - Changes in pigmentation suggesting atrophy
                    
                    4. **Tessellated Fundus**:
                       - Tigroid or "tiger-striped" appearance
                       - Increased visibility of choroidal vessels
                    
                    **Note**: This analysis is computer-generated and should be verified by a qualified eye care professional.
                    """)
                
                # Option to save the heatmap
                heatmap_img_base64 = image_to_base64(heatmap_image)
                href = f'<a href="data:image/png;base64,{heatmap_img_base64}" download="fundus_heatmap.png">Download Heatmap Image</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try a different image or check if the uploaded file is a valid fundus image.")
    
    # Option to analyze previous images from session
    st.markdown("### Or analyze a previously uploaded image")
    
    # Check if we have previous predictions
    if (hasattr(st.session_state, 'prediction_history') and 
        st.session_state.prediction_history and 
        len(st.session_state.prediction_history) > 0):
        
        # Create a list of previous images
        image_options = ["Select an image"] + [p['image_name'] for p in st.session_state.prediction_history]
        
        # Create a dropdown to select an image
        selected_image = st.selectbox("Select a previously analyzed image:", image_options)
        
        if selected_image != "Select an image":
            st.info(f"To analyze '{selected_image}', please re-upload the image file.")
            st.markdown("Previous predictions are tracked, but the image data itself isn't stored for privacy reasons.")
    else:
        st.info("No previously analyzed images found. Please upload an image to begin analysis.")
    
    # Educational information about myopia
    st.markdown("---")
    with st.expander("Learn more about myopia risk factors"):
        st.markdown("""
        ### Risk Factors for Myopia
        
        **Fundus Signs of Myopia:**
        
        1. **Optic Disc Changes**
           - Tilted disc appearance
           - Peripapillary atrophy (crescent-shaped thinning around the optic disc)
           - Larger optic disc size
        
        2. **Vascular Changes**
           - Straightening of retinal vessels
           - Attenuation (thinning) of blood vessels
           - Changes in vessel branching patterns
        
        3. **Retinal Changes**
           - Tessellated fundus (visible choroidal vessels giving a "tiger stripe" appearance)
           - Peripheral retinal thinning
           - Temporal dragging of retinal vessels
        
        4. **Advanced Signs (pathological myopia)**
           - Lacquer cracks (breaks in Bruch's membrane)
           - Posterior staphyloma (outward bulging of the back of the eye)
           - Fuchs' spot (pigmentation in the macula)
        
        **Clinical Management of Myopia:**
        
        - Regular comprehensive eye examinations
        - Optical correction (glasses, contact lenses)
        - Myopia control interventions (specialized lenses, atropine therapy)
        - Monitoring for complications (retinal detachment, glaucoma, cataracts)
        """)

def add_heatmap_to_navbar():
    """
    Update the navbar to include the heatmap visualization option
    """
    # This function doesn't directly modify files
    # It returns a modified version of the navbar options
    return ["Dashboard", "Upload & Predict", "Visualizations", "Heatmap Analysis", "Database Management", "Model Training"]