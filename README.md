# DiRetina Dashboard

DiRetina is an AI-powered healthcare dashboard for myopia detection using fundus image analysis. The system analyzes retinal fundus images to detect signs of myopia and provides healthcare professionals with visualization and analytics tools.

## Features

- TensorFlow-based AI model for fundus image analysis with fallback mechanisms
- Advanced visualization of eye disease metrics and trends
- Patient data management
- Training capabilities for model improvement
- Database integration (PostgreSQL or SQLite)

## Running Locally with SQLite

For local development and testing, you can run DiRetina with a SQLite database instead of PostgreSQL. This is simpler and doesn't require setting up a database server.

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or download the repository**

2. **Set up a virtual environment** (recommended)
   ```
   python -m venv diretina-env
   ```

3. **Activate the virtual environment**
   - Windows:
     ```
     diretina-env\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source diretina-env/bin/activate
     ```

4. **Install dependencies**
   ```
   pip install streamlit opencv-python pandas sqlalchemy plotly pillow numpy
   ```

   For TensorFlow (optional):
   ```
   pip install tensorflow
   ```

5. **Run the local version of the app**
   ```
   streamlit run local_app.py
   ```

   This will start the app with SQLite database support. Your data will be stored in `diretina.db` in the application directory.

6. **Access the dashboard**
   Open your browser and go to http://localhost:8501

## Running on Replit

The application can also be run directly on Replit, which will use PostgreSQL for database storage.

1. Click the "Run" button in Replit
2. Access the app in the webview at port 5000

## Project Structure

- **app.py**: Main application file (PostgreSQL version)
- **local_app.py**: Local application file (SQLite version)
- **components/**: UI components for different sections
  - dashboard.py: Main dashboard view
  - upload.py: Image upload and prediction section
  - visualization.py: Data visualization section
  - database_mgmt.py: Database management section
- **utils/**: Helper modules
  - database.py: Database connection and operations (PostgreSQL)
  - local_database.py: Local database operations (SQLite)
  - image_processing.py: Image preprocessing functions
  - model.py: Main model interface
  - tf_model.py: TensorFlow model implementation

## AI Model Information

The application uses a dual-approach model:

1. TensorFlow Model: Uses MobileNetV2 architecture for transfer learning
2. Fallback Model: Uses traditional image processing to analyze fundus characteristics:
   - Optic disc detection
   - Blood vessel pattern analysis
   - Peripheral region analysis
   - Tigroid pattern detection

## PostgreSQL vs SQLite

- **PostgreSQL**: Used in the main application (app.py). Better for production and multi-user scenarios.
- **SQLite**: Used in the local version (local_app.py). Simpler for local development and testing.

## License

Copyright © 2024