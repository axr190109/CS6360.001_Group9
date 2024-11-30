SQL Query Runtime Prediction Project

**Overview**
This project implements a machine learning-based solution to predict SQL query runtime, result size, and error class. It replicates the methodology outlined in a research paper, using data-driven models and a Flask web interface for interaction.

**Repository Contents**
models/: Contains the trained models and scripts (train_model.py, predict.py) for training and evaluation.
data/: Contains datasets like extracted_features.csv used for training and evaluation.
app/: Includes the Flask application and templates/ folder for the web interface.
README.md: This file, outlining the installation, usage, and experiment steps.
Installation Instructions
Follow these steps to set up the project on your local machine:

**Prerequisites**
Install Python (3.12.6)
Download Python:

Go to the official Python website: https://www.python.org/downloads/.
Select Python 3.12.6 and download the appropriate installer for your operating system.
Run the Installer:

Open the downloaded installer file.
Check the box that says Add Python 3.12 to PATH before proceeding.
Click on Customize Installation and ensure all optional features are selected.
Click Install Now.

**Verify Installation:**
Open a terminal or command prompt and type:
python --version
Ensure it shows Python 3.12.6.
Install pip (if not already installed):

pip is included with Python 3.12.6, but if needed, run:
python -m ensurepip --upgrade

**Setup Steps**
Clone the repository:
git clone https://github.com/axr190109/CS6360.001_Group9.git
cd CS6360.001_Group9

**Create a virtual environment (recommended):**
python -m venv venv
source venv/bin/activate       # On Mac/Linux
venv\Scripts\activate          # On Windows

**Install dependencies:**
pip install -r requirements.txt

**Install scikit-learn (if not included in requirements.txt):**
pip install scikit-learn

**Ensure TensorFlow is correctly installed:**
pip install tensorflow

**Running the Application and Scripts**
Follow these steps to execute the project components:

**Step 1: Train Models (Optional if pre-trained models are included)**
To train the models from scratch:
Navigate to the models/ directory:
cd models

Run the training script:
python train_model.py
This will generate runtime_model.keras, size_model.keras, and error_model.keras in the models/ directory.

**Step 2: Predict Using the Models**
To test the trained models:
Ensure you are in the models/ directory:
cd models
Run the prediction script:
python predict.py
The script generates example inputs and outputs the predicted runtime, result size, and error class.

**Step 3: Start the Flask Application**
To launch the web interface:
Navigate back to the root directory of the project:
cd ..
Run the Flask app:
python app.py
Open a web browser and go to http://127.0.0.1:5000.

**Step 4: Use the Web Interface**
Enter SQL query parameters (e.g., number of joins, query length) in the form.
Submit the form to view the predicted runtime, result size, and error class.

**Experiment Procedure**
The experiments in this project evaluate the accuracy of the machine learning models. Follow these steps:

**Step 1: Input Dataset**
The dataset used for training and evaluation is located at data/extracted_features.csv.

**Step 2: Run Model Evaluation**
Navigate to the models/ directory:
cd models
Run the training script:
python train_model.py
The script outputs metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and classification accuracy for each model.

**Step 3: Interpret Results**
The Runtime Model predicts the runtime of SQL queries in seconds.
The Size Model predicts the result size of SQL queries.
The Error Class Model predicts the error category (0, 1, or 2).

**Notes for the TA**
Ensure Python, TensorFlow, and scikit-learn are installed before running the project.
Pre-trained models are provided in the models/ directory; training is optional for demonstration purposes.
The Flask app provides an interactive interface for testing.
For any issues, refer to the models/ and app/ directories for troubleshooting steps or contact the group.






