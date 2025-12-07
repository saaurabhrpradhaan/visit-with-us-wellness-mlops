# visit-with-us-wellness-mlops

This project builds an end‑to‑end MLOps pipeline to predict whether a customer will buy the Wellness Tourism Package for the company “Visit with Us”.

## Project Goal

The goal is to build a classification model that predicts the target `ProdTaken`.  
The model uses customer details and interaction data.  
The pipeline covers data cleaning, feature engineering, model training and tuning, saving the best model to the Hugging Face Hub, and deploying a Streamlit app for prediction.

## Repository Structure

- `data/`
  - `raw/tourism.csv` – original dataset
  - `processed/train.csv` and `processed/test.csv` – cleaned and split data
- `src/`
  - `data_prep.py` – loads raw data, cleans it, creates features, and saves train/test files
  - `train.py` – trains a Random Forest with GridSearchCV, evaluates ROC‑AUC, logs results, and saves the best model
- `model/`
  - `wellness_rf.skops` – saved model pipeline
- `experiments/`
  - `experiments_log.csv` – log of the best parameters and ROC‑AUC score
- `app/`
  - `app_streamlit.py` – Streamlit web app for single‑customer predictions
- `hf_deploy/`
  - `requirements.txt` – Python dependencies for deployment
- `.github/workflows/`
  - `pipeline.yml` – GitHub Actions workflow that runs data prep and training when code is pushed to `main`

## How to Run

1. Clone the repository  
   `git clone https://github.com/saaurabhrpradhaan/visit-with-us-wellness-mlops.git`

2. Install dependencies  
   `pip install -r hf_deploy/requirements.txt`

3. Run data preparation  
   `python src/data_prep.py`

4. Train the model and push it to Hugging Face  
   `python src/train.py`

5. (Optional) Run the Streamlit app locally  
   `streamlit run app/app_streamlit.py`

   
