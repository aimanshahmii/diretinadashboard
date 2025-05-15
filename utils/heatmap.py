import numpy as np
import cv2
import io
import base64
from PIL import Image

def generate_heatmap(image_array, prediction_confidence=0.5):
    """
    Generate a heatmap highlighting high-risk areas in fundus images.
    This simulates attention areas that might indicate myopia or other eye conditions.
    
    Args:
        image_array: The fundus image as a numpy array
        prediction_confidence: The confidence score from the prediction model
        
    Returns:
        heatmap_img: PIL Image with the heatmap overlay
        risk_areas: List of (x, y, radius, risk_score) tuples for high-risk areas
    """
    # Create a copy to avoid modifying the original
    original_img = image_array.copy()
    
    # Ensure image is in RGB format (for color-based processing)
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    elif original_img.shape[2] == 4:
        original_img = original_img[:, :, :3]  # Remove alpha channel
    
    # Convert to proper dimensions if needed
    height, width = original_img.shape[:2]
    
    # Create empty heatmap with same dimensions as the original image
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Identify risk zones based on image analysis
    risk_areas = []
    
    # 1. Detect the optic disc (usually appears as a bright circular area)
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find the brightest region (potential optic disc)
    disc_mask = cv2.threshold(blurred, np.percentile(blurred, 90), 255, cv2.THRESH_BINARY)[1]
    disc_mask = disc_mask.astype(np.uint8)
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Optic disc coordinates (default to center if not found)
    disc_x, disc_y = width // 2, height // 2
    disc_radius = min(width, height) // 10
    
    if contours:
        # Find the largest contour (likely the optic disc)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
            # Get the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                disc_x = int(M["m10"] / M["m00"])
                disc_y = int(M["m01"] / M["m00"])
                
                # Estimate the radius from contour area
                disc_area = cv2.contourArea(largest_contour)
                disc_radius = int(np.sqrt(disc_area / np.pi))
    
    # Add optic disc to risk areas with moderate risk
    # The optic disc itself is not a risk, but changes around it can be
    optic_disc_risk = 0.3 * prediction_confidence
    risk_areas.append((disc_x, disc_y, disc_radius, optic_disc_risk))
    
    # 2. Detect blood vessel patterns
    # Apply CLAHE to enhance vessel contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Get vessel-like structures using edge detection
    edges = cv2.Canny(enhanced, 30, 70)
    
    # Clean up with morphology
    vessel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessels = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vessel_kernel)
    
    # Find vessel density in different regions
    # Divide the image into a grid and assess vessel density in each cell
    grid_size = 4
    cell_h, cell_w = height // grid_size, width // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Cell coordinates
            x1, y1 = j * cell_w, i * cell_h
            x2, y2 = (j + 1) * cell_w, (i + 1) * cell_h
            
            # Skip cells that are very close to the center (optic disc)
            cell_center_x = (x1 + x2) // 2
            cell_center_y = (y1 + y2) // 2
            distance_to_disc = np.sqrt((cell_center_x - disc_x)**2 + (cell_center_y - disc_y)**2)
            
            if distance_to_disc < disc_radius * 1.5:
                continue
                
            # Calculate vessel density in this cell
            cell_vessels = vessels[y1:y2, x1:x2]
            vessel_density = np.sum(cell_vessels > 0) / (cell_w * cell_h)
            
            # High vessel density variations can indicate issues
            if vessel_density > 0.15:  # Threshold for abnormal vessel density
                risk_score = min(0.7, vessel_density * 2) * prediction_confidence
                risk_areas.append((
                    (x1 + x2) // 2,  # center x
                    (y1 + y2) // 2,  # center y
                    cell_w // 3,     # radius (smaller than the cell)
                    risk_score
                ))
    
    # 3. Analyze peripheral regions for thinning
    # Create a mask for the periphery region
    margin = int(min(width, height) * 0.1)
    center_x, center_y = width // 2, height // 2
    max_radius = min(width, height) // 2 - margin
    
    # Check several points in the periphery
    num_samples = 8
    for angle in np.linspace(0, 2*np.pi, num_samples, endpoint=False):
        # Point on the periphery
        px = int(center_x + np.cos(angle) * max_radius)
        py = int(center_y + np.sin(angle) * max_radius)
        
        # Ensure within bounds
        px = max(0, min(px, width-1))
        py = max(0, min(py, height-1))
        
        # Get a small region around this point
        region_size = max_radius // 4
        x1 = max(0, px - region_size)
        y1 = max(0, py - region_size)
        x2 = min(width, px + region_size)
        y2 = min(height, py + region_size)
        
        # Analyze the periphery region color and texture
        region = original_img[y1:y2, x1:x2]
        
        if region.size > 0:
            # Calculate various metrics that might indicate thinning or abnormalities
            # Brightness in the periphery (higher can indicate thinning)
            brightness = np.mean(region)
            normalized_brightness = brightness / 255.0
            
            # Higher contrast might indicate abnormal patterns
            contrast = np.std(region) / 255.0
            
            # Color variations (red channel often shows retinal thinning better)
            if region.shape[2] >= 3:
                red_dominance = np.mean(region[:,:,0]) / (np.mean(region[:,:,1]) + np.mean(region[:,:,2]) + 1e-6)
                
                # Combine metrics to estimate risk
                risk_metrics = [
                    normalized_brightness * 0.4,  # Brightness contribution
                    contrast * 0.3,              # Contrast contribution
                    (red_dominance - 0.5) * 0.3  # Color contribution
                ]
                
                risk_score = sum(risk_metrics) * prediction_confidence
                
                # Add risk area if the score is significant
                if risk_score > 0.2:
                    risk_areas.append((px, py, region_size, risk_score))
    
    # Generate the actual heatmap
    for x, y, radius, risk in risk_areas:
        # Create a small gaussian "hotspot" at the risk location
        for i in range(max(0, y - radius * 2), min(height, y + radius * 2)):
            for j in range(max(0, x - radius * 2), min(width, x + radius * 2)):
                # Distance to the center of the risk area
                distance = np.sqrt((i - y)**2 + (j - x)**2)
                # Gaussian falloff
                if distance < radius * 2:
                    # Normalize the distance to the radius
                    normalized_dist = distance / (radius * 2)
                    # Gaussian function for smooth falloff
                    heat_value = risk * np.exp(-4 * normalized_dist**2)
                    # Update the heatmap with maximum value
                    heatmap[i, j] = max(heatmap[i, j], heat_value)
    
    # Normalize heatmap values to 0-1 range
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Convert heatmap to colormap (jet is commonly used for heatmaps)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend the heatmap with original image
    alpha = 0.5  # Transparency factor
    beta = 1 - alpha
    
    # Resize heatmap_colored to match original_img if needed
    if heatmap_colored.shape[:2] != original_img.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (original_img.shape[1], original_img.shape[0]))
    
    # Ensure the images have the same number of channels
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    if heatmap_colored.shape[2] == 1:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2BGR)
    
    # Create the blended image
    blended = cv2.addWeighted(original_img, beta, heatmap_colored, alpha, 0)
    
    # Convert to PIL Image for Streamlit
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(blended_rgb)
    
    return heatmap_img, risk_areas

def get_risk_analysis(risk_areas):
    """
    Generate a textual analysis of the risk areas
    
    Args:
        risk_areas: List of (x, y, radius, risk_score) tuples
        
    Returns:
        String with risk analysis
    """
    if not risk_areas:
        return "No significant risk areas detected."
    
    # Sort risk areas by risk score (highest first)
    sorted_risks = sorted(risk_areas, key=lambda x: x[3], reverse=True)
    
    # Count risk areas by severity
    high_risk = sum(1 for _, _, _, score in sorted_risks if score > 0.6)
    medium_risk = sum(1 for _, _, _, score in sorted_risks if 0.3 < score <= 0.6)
    low_risk = sum(1 for _, _, _, score in sorted_risks if score <= 0.3)
    
    analysis = []
    
    # Overall summary
    if high_risk > 0:
        analysis.append(f"Detected {high_risk} high-risk areas that may indicate significant myopic changes.")
    if medium_risk > 0:
        analysis.append(f"Found {medium_risk} medium-risk areas with potential early signs of myopia.")
    if low_risk > 0:
        analysis.append(f"Observed {low_risk} low-risk areas that should be monitored in future examinations.")
    
    # Add details about the highest risk areas (up to 3)
    if sorted_risks:
        analysis.append("\nHighest risk areas:")
        
        for i, (_, _, _, score) in enumerate(sorted_risks[:3]):
            severity = "High" if score > 0.6 else "Medium" if score > 0.3 else "Low"
            confidence = f"{score:.1%}"
            analysis.append(f"• Area {i+1}: {severity} risk ({confidence} confidence)")
    
    # Recommendations
    analysis.append("\nRecommendations:")
    if high_risk > 0:
        analysis.append("• Prompt referral to an ophthalmologist for detailed examination")
        analysis.append("• Consider specialized imaging such as OCT for structural analysis")
    elif medium_risk > 0:
        analysis.append("• Follow-up examination within 3-6 months")
        analysis.append("• Monitor for progression of myopic changes")
    else:
        analysis.append("• Routine follow-up as per standard guidelines")
        analysis.append("• Patient education on maintaining eye health")
    
    return "\n".join(analysis)

def image_to_base64(image):
    """
    Convert a PIL image to base64 string
    
    Args:
        image: PIL Image object
        
    Returns:
        base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str