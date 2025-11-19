# DiRetina Dashboard

## Overview

DiRetina is an AI-powered healthcare dashboard designed for myopia detection using fundus (retinal) image analysis. The application allows healthcare professionals to upload fundus images, receive AI-based diagnostic predictions, and visualize detection trends and patterns. The system features both TensorFlow-based deep learning models and traditional image processing fallback mechanisms to ensure reliable operation across different environments.

The application is built with Streamlit for the web interface, providing an interactive dashboard for patient data management, image analysis, and visual reporting of eye health metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit-based web application
- **Multi-page structure**: Organized into modular components (dashboard, upload, visualization, heatmap view, database management)
- **Session state management**: Tracks uploaded images, predictions, model state, and processed files across user sessions
- **Responsive layout**: Uses Streamlit's column-based layout system for metrics cards and visualizations
- **Real-time updates**: Interactive charts and metrics that update as new images are analyzed

**Key UI Components**:
1. **Dashboard**: Summary statistics and key metrics (images analyzed, myopia detection count, eye health score)
2. **Upload & Predict**: File upload interface with support for multiple image formats (JPG, PNG)
3. **Visualizations**: Plotly-based interactive charts showing detection trends over time
4. **Heatmap View**: Visual representation of high-risk zones in fundus images

**UI Design Philosophy**:
- Clean, professional interface designed for presentation and deployment
- Removed sample images and placeholder content for production-ready appearance
- Focused on essential functionality without visual clutter
- Streamlined navigation and information architecture

### Backend Architecture

**Application Entry Points**:
- `app.py`: Main application for cloud/production deployment with PostgreSQL support
- `local_app.py`: Local development version with SQLite/MySQL fallback

**AI/ML Model Strategy**:
The application implements a **dual-model architecture** with automatic fallback:

1. **Primary**: TensorFlow-based deep learning model (`TensorFlowFundusModel`)
   - Uses MobileNetV2 architecture with transfer learning
   - Pre-trained on ImageNet, fine-tuned for fundus image classification
   - Binary classification: Myopia (1) vs Normal (0)
   - Input: 224x224x3 RGB images

2. **Fallback**: Traditional image processing model (`FundusAnalyzer`)
   - OpenCV-based fundus image analysis
   - Detects clinical indicators: optic disc size, vessel tortuosity, retinal curvature
   - Rule-based scoring system for myopia likelihood
   - Ensures system functionality when TensorFlow is unavailable

**Model Selection Logic**:
- Checks TensorFlow availability at runtime using `importlib.util.find_spec`
- Gracefully degrades to traditional analyzer if TensorFlow import fails
- Maintains consistent prediction interface across both model types

**Image Processing Pipeline**:
1. Image validation (checks for fundus-like characteristics)
2. Preprocessing: RGB conversion, resizing to 224x224, normalization to [0,1]
3. Prediction generation with confidence scores
4. Heatmap generation for risk zone visualization

**Session Management**:
- Version tracking to detect application restarts
- File deduplication using file ID (name + size)
- Persistent prediction history within session
- Automatic reset of processed files on app restart

### Data Storage Solutions

**Database Abstraction Layer**:
The application supports multiple database backends through SQLAlchemy ORM:

1. **PostgreSQL** (Production): Configured via `DATABASE_URL` environment variable
   - Supports Supabase pooler URLs with automatic URL fixing
   - Connection string validation and encoding for special characters

2. **SQLite** (Local Development): File-based database (`diretina.db`)
   - Zero-configuration option for testing
   - Automatic fallback if PostgreSQL unavailable

3. **MySQL** (Alternative): Configurable through `local_database.py`
   - Supports MySQL Workbench integration
   - Configured via `MYSQL_CONFIG` dictionary

**Database Schema**:
Two primary entities using SQLAlchemy declarative base:

1. **Prediction Table**:
   - id (Integer, Primary Key)
   - patient_id (Integer, optional)
   - image_filename (String)
   - prediction (Integer: 0=Normal, 1=Myopia)
   - confidence (Float)
   - timestamp (DateTime)
   - analysis_details (Text, JSON-serializable)

2. **Patient Table**:
   - id (Integer, Primary Key)
   - patient_name (String)
   - age (Integer, optional)
   - gender (String, optional)
   - contact (String, optional)
   - created_at (DateTime)

**Database Operations**:
- Cached database initialization using `@st.cache_resource`
- Automatic table creation on first connection
- Connection health checks with user feedback
- Batch retrieval of predictions and patient records for dashboard metrics

### Authentication and Authorization

**Current State**: No authentication implemented
- Open access to all features
- Suitable for internal/development use
- Future enhancement opportunity for multi-user scenarios

**Considerations for Production**:
- Would require user authentication layer (e.g., Streamlit-authenticator, OAuth)
- Role-based access control for different user types (doctors, admins, researchers)
- Patient data privacy compliance (HIPAA, GDPR considerations)

### Component Organization

**Modular Structure**:
```
components/
├── dashboard.py          # Main dashboard with summary metrics
├── upload.py            # Image upload and prediction interface
├── visualization.py     # Detailed charts and trend analysis
├── heatmap_view.py      # Risk zone visualization
└── database_mgmt.py     # Database viewing and management
```

**Utilities**:
```
utils/
├── model.py             # Model loading and prediction logic
├── tf_model.py          # TensorFlow model implementation
├── image_processing.py  # Image preprocessing and validation
├── heatmap.py           # Heatmap generation algorithms
├── database.py          # PostgreSQL/SQLite database operations
└── local_database.py    # MySQL database operations
```

## External Dependencies

### Core Frameworks
- **Streamlit**: Web application framework and UI components
- **SQLAlchemy**: ORM for database abstraction and operations
- **Pandas**: Data manipulation and DataFrame operations for analytics
- **NumPy**: Numerical computing and array operations

### Machine Learning & Image Processing
- **TensorFlow** (optional): Deep learning framework for AI model
  - Keras API for model building (Sequential, MobileNetV2)
  - Pre-trained ImageNet weights for transfer learning
- **OpenCV (cv2)**: Image processing operations (resize, color conversion, morphological operations)
- **Pillow (PIL)**: Image loading and basic manipulation

### Visualization
- **Plotly Express & Graph Objects**: Interactive charts (line charts, bar charts, pie charts)
- **Plotly.graph_objects**: Advanced heatmap overlays and custom visualizations

### Database Drivers
- **psycopg2** (implied): PostgreSQL adapter for SQLAlchemy
- **PyMySQL** (optional): MySQL connector for local development
- **SQLite3** (built-in): Default Python SQLite support

### Third-Party Services
- **Supabase** (optional): PostgreSQL-compatible database hosting
  - Connection pooler support on port 6543
  - Automatic URL parsing and credential handling
  - Environment variable configuration via `DATABASE_URL`

### Development & Configuration
- **Semgrep**: Static analysis for security rule checking (Bicep-focused rules for Azure deployment)
- **Environment Variables**: Database connection strings, TensorFlow logging configuration

### Key Technical Decisions

1. **Dual-model approach**: Ensures system reliability across environments with varying TensorFlow support
2. **Database flexibility**: Multi-backend support allows deployment flexibility from local testing to cloud production
3. **Session-based state**: Streamlit session state for maintaining user context without traditional backend sessions
4. **Modular component design**: Separation of concerns allows independent development and testing of features
5. **Fallback mechanisms**: Multiple layers of graceful degradation (TensorFlow → traditional model, PostgreSQL → SQLite)