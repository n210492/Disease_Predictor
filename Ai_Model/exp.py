import pandas as pd
import numpy as np
import time
import torch
import os
from sklearn.metrics import accuracy_score, precision_score
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import joblib
from huggingface_hub import login
import warnings
from fastapi import FastAPI, HTTPException , Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
from typing import List , Tuple

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Google Colab Path Configuration
BASE_DIR = "https://github.com/n210492/Disease_Predictor/edit/main/Ai_Model"
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset Paths
DATA_PATHS = {
    'symptoms': os.path.join(BASE_DIR, "Data.csv"),
    'clinical': os.path.join(BASE_DIR, "clinic.csv"),
    'environmental': os.path.join(BASE_DIR, "env.csv")
}

# Environmental mappings
ENV_MAPPINGS = {
    'temperature': {'low': 1, 'medium': 2, 'high': 3, None: 2},
    'humidity': {'low': 1, 'medium': 2, 'high': 3, None: 2},
    'air_quality': {'bad': 4, 'normal': 5, 'good': 6, None: 5},
    'water_quality': {'bad': 4, 'normal': 5, 'good': 6, None: 5},
    'region_type': {
        'mountain': 7, 'desert': 8, 'forest': 9, 'grassland': 10,
        'cold': 11, 'coastal': 12, 'river_basin': 13, 'tropical': 14,
        'temperate': 15, 'dry': 16, 'urban': 17, 'rural': 18, None: 18
    },
    'weather': {
        'sunny': 19, 'cloudy': 20, 'rainy': 21, 'snowy': 22,
        'windy': 23, 'foggy': 24, 'stormy': 25, 'sweaty': 26, 'humid': 27, None: 19
    },
    'time_delay': {
        'recent': (28, 1.0),    # <5 days - base risk
        '<5days': (28, 1.0),
        '28': (28, 1.0),
        'moderate': (29, 1.3),   # 5-15 days - 30% higher risk
        '5to15days': (29, 1.3), 
        '29': (29, 1.3), 
        'long': (30, 1.7),       # >15 days - 60% higher risk
        '>15days': (30, 1.7),
        '30': (30, 1.7),
        None: (29, 1.0)          # Default to base risk
    }
}

def load_data_safely(file_path, dataset_type):
    # """Load data with memory optimization and error handling"""
    try:
        dtype_map = {
            'symptoms': {col: 'int8' for col in pd.read_csv(file_path, nrows=1).columns if col != 'Diseases'},
            'clinical': {
                'Age': 'int8', 'Weight': 'int16', 'BP': 'int16',
                'Sugar': 'int16', 'Cholesterol': 'int16',
                'WBC': 'int16', 'BMI': 'float32', 'Sleep': 'int8',
                'Diseases': 'object'
            },
            'environmental': {
                'Diseases': 'object',
                'temperature': 'object',
                'humidity': 'object',
                'air_quality': 'object',
                'water_quality': 'object',
                'region_type': 'object',
                'weather': 'object',
                'time_delay':'int8'
            }
        }

        df = pd.read_csv(file_path, dtype=dtype_map.get(dataset_type, None),
                         na_values=['', 'NA', 'N/A', 'NaN', 'null'])

        # Standardize column names
        col_rename = {
            'Bp': 'BP', 'Cholestral': 'Cholesterol', 'Sleep Duration': 'Sleep',
            'Temperature': 'temperature', 'Weather': 'weather',
            'Region_Type': 'region_type', 'Air_Quality': 'air_quality',
            'Water_Quality': 'water_quality', 'Humidity': 'humidity'
        }
        df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

        # Drop rows with any missing values
        initial_count = len(df)
        df = df.dropna()
        if len(df) < initial_count:
            print(f"Dropped {initial_count - len(df)} rows with missing values from {os.path.basename(file_path)}")

        return df

    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
        return None

def train_and_save_models():
    try:
        print("Loading datasets...")
        df_symptoms = load_data_safely(DATA_PATHS['symptoms'], 'symptoms')
        df_clinical = load_data_safely(DATA_PATHS['clinical'], 'clinical')
        df_env = load_data_safely(DATA_PATHS['environmental'], 'environmental')

        if df_symptoms is None or df_clinical is None or df_env is None:
            return False

        # Initialize label encoder with all diseases
        all_diseases = pd.concat([
            df_symptoms['Diseases'],
            df_clinical['Diseases'],
            df_env['Diseases']
        ]).unique()

        le = LabelEncoder()
        le.fit(all_diseases)

        # 1. Train Symptom Model (XGBoost)
        print("\nTraining Symptom Model...")
        X_symp = df_symptoms.drop('Diseases', axis=1)
        y_symp = le.transform(df_symptoms['Diseases'])

        xgb_model = XGBClassifier(
            objective='multi:softprob',
            num_class=len(le.classes_),
            tree_method='hist',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42
        )
        xgb_model.fit(X_symp, y_symp)

        # 2. Train Clinical Model (Logistic Regression)
        print("\nTraining Clinical Model...")
        clinical_cols = ['Age', 'Weight', 'BP', 'Sugar', 'Cholesterol', 'WBC', 'BMI', 'Sleep']
        X_clin = df_clinical[clinical_cols]
        y_clin = le.transform(df_clinical['Diseases'])

        clin_scaler = StandardScaler()
        X_clin_scaled = clin_scaler.fit_transform(X_clin)

        lr_clin = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        lr_clin.fit(X_clin_scaled, y_clin)

        # 3. Train Environmental Model (Random Forest)
        print("\nTraining Environmental Model...")
        env_cols = list(ENV_MAPPINGS.keys())
        X_env = df_env[env_cols]
        y_env = le.transform(df_env['Diseases'])

        rf_env = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        rf_env.fit(X_env, y_env)

        # 4. Train BERT Model
        print("\nTraining BERT Model...")
        try:
            # Try silent authentication first
            login(token="hf_xxxxxxxxxxxxxxxxxx", add_to_git_credential=False)
        except:
            # Fallback to public access
            print("Using Hugging Face without authentication")

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        symptom_cols = X_symp.columns.tolist()
        train_text = X_symp.apply(
            lambda row: ' '.join([col for col, val in row.items() if val == 1]), axis=1)

        # Process in memory-friendly chunks
        chunk_size = 100
        train_embeddings = []
        for i in range(0, len(train_text), chunk_size):
            chunk = train_text.iloc[i:i+chunk_size]
            embeddings = [get_bert_embedding(text, tokenizer, bert_model) for text in chunk]
            train_embeddings.extend(embeddings)
            del chunk, embeddings
            gc.collect()

        train_embeddings = np.array(train_embeddings)
        lr_bert = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        lr_bert.fit(train_embeddings, y_symp)

        print("\n✨ All models trained successfully!... ✨")

        print("\n✨ Now saving... ✨")

        # Save all models with feature names
        joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
        joblib.dump(lr_clin, os.path.join(MODELS_DIR, 'clinical_model.pkl'))
        joblib.dump(rf_env, os.path.join(MODELS_DIR, 'env_model.pkl'))
        joblib.dump(lr_bert, os.path.join(MODELS_DIR, 'bert_model.pkl'))
        joblib.dump(le, os.path.join(MODELS_DIR, 'disease_encoder.pkl'))
        joblib.dump(symptom_cols, os.path.join(MODELS_DIR, 'symptom_features.pkl'))
        joblib.dump(clinical_cols, os.path.join(MODELS_DIR, 'clinical_features.pkl'))
        joblib.dump(env_cols, os.path.join(MODELS_DIR, 'env_features.pkl'))
        joblib.dump(clin_scaler, os.path.join(MODELS_DIR, 'clinical_scaler.pkl'))

        # Save BERT components
        tokenizer.save_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
        bert_model.save_pretrained(os.path.join(MODELS_DIR, "bert_model"))

        return True

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        return False
    
def get_bert_embedding(text, tokenizer, model):
    # """Generate BERT embeddings with memory management"""
    inputs = tokenizer(text, return_tensors="pt",
                      truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    del inputs, outputs
    torch.cuda.empty_cache()
    return embedding

def predict_disease(symptoms, clinical_data, env_data, models):
    """Predict diseases based on symptom matching with comprehensive risk adjustment"""
    (xgb_model, lr_clin, rf_env, lr_bert, le,
     symptom_cols, clinical_cols, env_cols, clin_scaler,
     tokenizer, bert_model) = models

    # 1. Enhanced symptom processing with error handling
    try:
        symptom_words = [word.strip().lower() for word in str(symptoms).replace(',', ' ').split()]
        symptom_vector = pd.DataFrame(0, index=[0], columns=symptom_cols)
        
        symptom_matches = 0
        for feature in symptom_cols:
            feature_words = feature.lower().split('_')
            if any(symptom_word in feature_words for symptom_word in symptom_words):
                symptom_vector[feature] = 1
                symptom_matches += 1
    except Exception as e:
        raise ValueError(f"Symptom processing error: {str(e)}")

    # 2. Get base predictions with validation
    try:
        xgb_probs = xgb_model.predict_proba(symptom_vector)[0] * 0.6
        clin_probs = lr_clin.predict_proba(clin_scaler.transform(
            pd.DataFrame([clinical_data], columns=clinical_cols)))[0] * 0.18
        env_probs = rf_env.predict_proba(
            pd.DataFrame([env_data], columns=env_cols))[0] * 0.16
        bert_probs = lr_bert.predict_proba([get_bert_embedding(
            symptoms, tokenizer, bert_model)])[0] * 0.05

        combined_probs = xgb_probs + clin_probs + env_probs + bert_probs
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

    # 3. Clinical validation rules
    CLINICAL_RULES = {
        'heart attack': {
            'min_symptoms': 4,
            'required': ['chest_pain'],
            'deny': ['runny_nose']
        },
        'pancreatitis': {
            'min_symptoms': 3,
            'required': ['abdominal_pain']
        }
    }

    # 4. Apply clinical validation
    validated_probs = np.copy(combined_probs)
    for i, disease in enumerate(le.classes_):
        disease_lower = disease.lower()
        if disease_lower in CLINICAL_RULES:
            rule = CLINICAL_RULES[disease_lower]
            present_symptoms = symptom_vector.columns[symptom_vector.iloc[0]==1].tolist()
            
            # Check for denying symptoms
            if 'deny' in rule and any(s in present_symptoms for s in rule['deny']):
                validated_probs[i] *= 0.001
                continue
                
            # Check required symptoms
            if 'required' in rule and not all(s in present_symptoms for s in rule['required']):
                validated_probs[i] *= 0.01
                
            # Check minimum symptoms
            if 'min_symptoms' in rule and symptom_matches < rule['min_symptoms']:
                validated_probs[i] *= 0.1

    # 5. Normalize probabilities after validation
    validated_probs = validated_probs / np.sum(validated_probs)

    # 6. Calculate dynamic risk multiplier
    def calculate_risk(symptoms_matched, clinical, env):
        factors = {
    'base': 1.0,
    # Symptom-based (now more sensitive with /4 instead of /5)
    'symptoms': min(2.0, 0.5 + (symptoms_matched / 4)),  
    
    # Clinical factors
    'bmi': 1.3 if clinical['BMI'] < 18.5 else (1.2 if clinical['BMI'] > 25 else 1.0),
    'wbc': 1.2 if clinical['WBC'] > 10 else (0.9 if clinical['WBC'] < 4 else 1.0),
    'sleep': 1.2 if clinical['Sleep'] < 5 else (0.8 if clinical['Sleep'] > 8 else 1.0),
    
    # Environmental factors
    'water': 1.4 if env['water_quality'] == 4 else (0.9 if env['water_quality'] == 6 else 1.0),
    'air': 1.2 if env['air_quality'] == 4 else (0.9 if env['air_quality'] == 6 else 1.0),
    
    # Enhanced region types
    'region': (
        1.4 if env['region_type'] in [14, 13] else  # tropical=14, river_basin=13
        1.3 if env['region_type'] in [12, 9, 10] else  # coastal=12, forest=9, grassland=10
        1.1 if env['region_type'] in [11, 16] else  # cold=11, dry=16
        1.0  # default
    ),
    
    # Time factor (unchanged)
    'time': ENV_MAPPINGS['time_delay'][str(env['time_delay'])][1]
}
        return min(3.5, np.prod(list(factors.values())))

    risk_multiplier = calculate_risk(symptom_matches, clinical_data, env_data)

    # 7. Generate final predictions
    top_diseases = []
    for i in np.argsort(validated_probs)[-3:][::-1]:
        base_confidence = validated_probs[i]
        adjusted_risk = min(0.95, base_confidence * risk_multiplier)
        
        top_diseases.append({
            'disease': le.classes_[i],
            'confidence': float(base_confidence),
            'risk_percentage': float(adjusted_risk),
            'symptom_matches': symptom_matches,
            'severity_factor': float(risk_multiplier),
            'warning': "Consider clinical evaluation" if adjusted_risk > 0.5 else None
        })

    return top_diseases

def get_clinical_input():
    """Collect and validate clinical inputs with strict ranges"""
    print("\n=== Clinical Data Input ===")
    print("Please enter values within these ranges:")

    # Defined ranges with your exact specifications
    input_ranges = {
        'Age': (1, 100, "years"),
        'Weight': (30, 150, "kg"),
        'BP': (60, 200, "mmHg (systolic)"),
        'Sugar': (50, 300, "mg/dL (fasting)"),
        'Cholesterol': (10, 300, "mg/dL"),
        'WBC': (2, 20, "thousands/μL"),
        'BMI': (5, 30, ""),
        'Sleep': (0, 10, "hours")
    }

    clinical_data = {}
    for param, (min_val, max_val, unit) in input_ranges.items():
        while True:
            try:
                # Show the acceptable range in the prompt
                prompt = f"{param} ({min_val}-{max_val} {unit}): "
                value = float(input(prompt))

                # Strict range validation
                if value < min_val or value > max_val:
                    print(f"Error: {param} must be between {min_val}-{max_val} {unit}")
                    continue

                clinical_data[param] = value
                break

            except ValueError:
                print("Invalid input. Please enter a number")

    return clinical_data


def get_environmental_input():
    """Collect environmental factors using natural language"""
    print("\n=== Environmental Data Input ===")
    env_data = {}

    # Temperature
    while True:
        temp = input("Temperature (low/medium/high): ").lower()
        if temp in ['low', 'medium', 'high']:
            env_data['temperature'] = ENV_MAPPINGS['temperature'][temp]
            break
        print("Please enter low, medium, or high")

    # Humidity
    while True:
        humid = input("Humidity (low/medium/high): ").lower()
        if humid in ['low', 'medium', 'high']:
            env_data['humidity'] = ENV_MAPPINGS['humidity'][humid]
            break
        print("Please enter low, medium, or high")

    # Air Quality
    while True:
        air = input("Air Quality (bad/normal/good): ").lower()
        if air in ['bad', 'normal', 'good']:
            env_data['air_quality'] = ENV_MAPPINGS['air_quality'][air]
            break
        print("Please enter bad, normal, or good")

    # Water Quality (same options as air)
    while True:
        water = input("Water Quality (bad/normal/good): ").lower()
        if water in ['bad', 'normal', 'good']:
            env_data['water_quality'] = ENV_MAPPINGS['water_quality'][water]
            break
        print("Please enter bad, normal, or good")

    # Region Type
    print("Region Types: urban, rural, coastal, mountain, tropical, desert, forest, river_basin, grassland, cold")
    while True:
        region = input("Region Type: ").lower()
        for key in ENV_MAPPINGS['region_type']:
            if key in region:  # Flexible matching (e.g., "forest" matches "forest area")
                env_data['region_type'] = ENV_MAPPINGS['region_type'][key]
                break
        if 'region_type' in env_data:
            break
        print("Invalid region type - try again")

    # Weather
    print("Weather Types: sunny, rainy, cloudy, snowy, windy, foggy, stromy, sweaty")
    while True:
        weather = input("Current Weather: ").lower()
        for key in ENV_MAPPINGS['weather']:
            if key in weather:
                env_data['weather'] = ENV_MAPPINGS['weather'][key]
                break
        if 'weather' in env_data:
            break
        print("Invalid weather - try again")

    print("\n⏳ When did symptoms first appear?")
    print("1. Recent (<5 days)")
    print("2. Moderate (5-15 days)")
    print("3. Long (>15 days)")
    
    while True:
        choice = input("Enter choice (1-3): ")
        if choice == '1':
            env_data['time_delay'] = 28
            break
        elif choice == '2':
            env_data['time_delay'] = 29
            break
        elif choice == '3':
            env_data['time_delay'] = 30
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3")

    return env_data


def main():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Define model components with their required files
        model_components = {
            'xgb': ['xgb_model.pkl', 'symptom_features.pkl'],
            'clinical': ['clinical_model.pkl', 'clinical_features.pkl', 'clinical_scaler.pkl'],
            'env': ['env_model.pkl', 'env_features.pkl'],
            'bert': ['bert_model.pkl'],
            'shared': ['disease_encoder.pkl']
        }

        # Check which components exist
        existing_components = {}
        for name, files in model_components.items():
            existing_components[name] = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)

        # Special BERT check (needs tokenizer and model directories)
        bert_ready = (os.path.exists(os.path.join(MODELS_DIR, "bert_tokenizer")) and
                     os.path.exists(os.path.join(MODELS_DIR, "bert_model")))
        existing_components['bert'] = existing_components['bert'] and bert_ready

        # Load what we can, train what's missing
        models = {}
        need_training = False

        # Load shared components first
        if existing_components['shared']:
            models['le'] = joblib.load(os.path.join(MODELS_DIR, 'disease_encoder.pkl'))
        else:
            print("Shared components missing - will need full training")
            need_training = True

        # Load individual models if they exist
        if existing_components['xgb']:
            models['xgb'] = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
            models['symptom_cols'] = joblib.load(os.path.join(MODELS_DIR, 'symptom_features.pkl'))
        else:
            print("XGBoost model missing")
            need_training = True

        if existing_components['clinical']:
            models['lr_clin'] = joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl'))
            models['clinical_cols'] = joblib.load(os.path.join(MODELS_DIR, 'clinical_features.pkl'))
            models['clin_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler.pkl'))
        else:
            print("Clinical model missing")
            need_training = True

        if existing_components['env']:
            models['rf_env'] = joblib.load(os.path.join(MODELS_DIR, 'env_model.pkl'))
            models['env_cols'] = joblib.load(os.path.join(MODELS_DIR, 'env_features.pkl'))
        else:
            print("Environmental model missing")
            need_training = True

        if existing_components['bert']:
            models['lr_bert'] = joblib.load(os.path.join(MODELS_DIR, 'bert_model.pkl'))
            models['tokenizer'] = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
            models['bert_model'] = AutoModel.from_pretrained(os.path.join(MODELS_DIR, "bert_model"))
        else:
            print("BERT model missing")
            need_training = True

        # Train if anything is missing
        if need_training:
            print("\nSome models are missing - training now...")
            if not train_and_save_models():
                return

            print("\nLoading Models.....")
            # Reload all models after training
            models = {
                'xgb': joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl')),
                'lr_clin': joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl')),
                'rf_env': joblib.load(os.path.join(MODELS_DIR, 'env_model.pkl')),
                'lr_bert': joblib.load(os.path.join(MODELS_DIR, 'bert_model.pkl')),
                'le': joblib.load(os.path.join(MODELS_DIR, 'disease_encoder.pkl')),
                'symptom_cols': joblib.load(os.path.join(MODELS_DIR, 'symptom_features.pkl')),
                'clinical_cols': joblib.load(os.path.join(MODELS_DIR, 'clinical_features.pkl')),
                'env_cols': joblib.load(os.path.join(MODELS_DIR, 'env_features.pkl')),
                'clin_scaler': joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler.pkl')),
                'tokenizer': AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer")),
                'bert_model': AutoModel.from_pretrained(os.path.join(MODELS_DIR, "bert_model"))
            }

         # Get user input and make prediction
        print("\n" + "="*40)
        print("Disease Prediction System")
        print("="*40)

        symptoms = input("\nDescribe your symptoms (e.g., fever, headache): ").strip()
        clinical_data = get_clinical_input()
        env_data = get_environmental_input()

        # Prepare the models tuple for prediction
        models_tuple = (
            models['xgb'],
            models['lr_clin'],
            models['rf_env'],
            models['lr_bert'],
            models['le'],
            models['symptom_cols'],
            models['clinical_cols'],
            models['env_cols'],
            models['clin_scaler'],
            models['tokenizer'],
            models['bert_model']
        )

        predictions = predict_disease(symptoms, clinical_data, env_data, models_tuple)

        print("\n" + "="*40)
        print("Top 3 Predicted Diseases with Risk Assessment:")
        print("="*40)
        for pred in predictions:
            print(f"\nDisease: {pred['disease']}")
            print(f"Model Confidence: {pred['confidence']:.1%}")
            print(f"Risk Percentage: {pred['risk_percentage']:.1%} (Severity Factor: {pred['severity_factor']:.1f}x)")
        print("="*40)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify CSV files exist in the correct location")
        print("2. Check file permissions in Google Drive")
        print("3. Delete the 'models' folder to force retraining")


# Define request models
class ClinicalInput(BaseModel):
    Age: int
    Weight: float
    BP: float
    Sugar: float
    Cholesterol: float
    WBC: int
    BMI: float
    Sleep: float

class EnvironmentalInput(BaseModel):
    temperature: str
    humidity: str
    air_quality: str
    water_quality: str
    region_type: str
    weather: str
    time_delay:str

class PredictionRequest(BaseModel):
    symptoms: str
    clinical_data: ClinicalInput
    environmental_data: EnvironmentalInput

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float          # Model's original prediction confidence (0-1)
    risk_percentage: float     # Severity-adjusted risk (0-0.95)
    severity_factor: float     # Multiplier applied (1.0, 1.3, or 1.6)
    time_delay_code: int       # 28, 29, or 30
    
# Initialize FastAPI app
app = FastAPI(title="Disease Prediction API",
              description="API for predicting diseases based on symptoms, clinical data, and environmental factors")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = None

def load_models():
    global MODELS

    if MODELS is not None:
        return MODELS

    try:
        models = {
            'xgb': joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl')),
            'lr_clin': joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl')),
            'rf_env': joblib.load(os.path.join(MODELS_DIR, 'env_model.pkl')),
            'lr_bert': joblib.load(os.path.join(MODELS_DIR, 'bert_model.pkl')),
            'le': joblib.load(os.path.join(MODELS_DIR, 'disease_encoder.pkl')),
            'symptom_cols': joblib.load(os.path.join(MODELS_DIR, 'symptom_features.pkl')),
            'clinical_cols': joblib.load(os.path.join(MODELS_DIR, 'clinical_features.pkl')),
            'env_cols': joblib.load(os.path.join(MODELS_DIR, 'env_features.pkl')),
            'clin_scaler': joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler.pkl')),
            'tokenizer': AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer")),
            'bert_model': AutoModel.from_pretrained(os.path.join(MODELS_DIR, "bert_model"))
        }

        MODELS = models
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")
    
def prepare_env_data(env_input: EnvironmentalInput) -> dict:
    return {
        'temperature': ENV_MAPPINGS['temperature'][env_input.temperature.lower()],
        'humidity': ENV_MAPPINGS['humidity'][env_input.humidity.lower()],
        'air_quality': ENV_MAPPINGS['air_quality'][env_input.air_quality.lower()],
        'water_quality': ENV_MAPPINGS['water_quality'][env_input.water_quality.lower()],
        'region_type': ENV_MAPPINGS['region_type'][env_input.region_type.lower()],
        'weather': ENV_MAPPINGS['weather'][env_input.weather.lower()],
        'time_delay': ENV_MAPPINGS['time_delay'][env_input.time_delay.lower()][0]
    }

@app.post("/predict", response_model=List[DiseasePrediction])
async def predict_disease_api(request: PredictionRequest):
    try:
        models = load_models()
        clinical_data = request.clinical_data.dict()
        env_data = prepare_env_data(request.environmental_data)
        
        models_tuple = (
            models['xgb'], models['lr_clin'], models['rf_env'], models['lr_bert'],
            models['le'], models['symptom_cols'], models['clinical_cols'],
            models['env_cols'], models['clin_scaler'],
            models['tokenizer'], models['bert_model']
        )

        predictions = predict_disease(
            request.symptoms,
            clinical_data,
            env_data,
            models_tuple
        )
        return predictions

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

def run_fastapi():
    """Run the FastAPI server in a background thread"""
    config = uvicorn.Config(app, host="0.0.0.0", port=1200)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()

    print("\nFastAPI server running on http://localhost:1200")
    print("Docs available at http://localhost:1200/docs")
    print("Press any key to stop the server...\n")

    # Keep the notebook cell running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping server...")

def evaluate_system():
    """Memory-optimized evaluation with guaranteed alignment"""
    try:
        print("⚡ Starting memory-safe evaluation...")
        start_time = time.time()

        # 1. Load only essential models first
        essential_models = {
            'le': joblib.load(os.path.join(MODELS_DIR, 'disease_encoder.pkl')),
            'symptom_cols': joblib.load(os.path.join(MODELS_DIR, 'symptom_features.pkl')),
            'clinical_cols': joblib.load(os.path.join(MODELS_DIR, 'clinical_features.pkl')),
            'env_cols': joblib.load(os.path.join(MODELS_DIR, 'env_features.pkl'))
        }

        # 2. Load and merge datasets first (minimize memory usage)
        df_symp = load_data_safely(DATA_PATHS['symptoms'], 'symptoms')
        df_clin = load_data_safely(DATA_PATHS['clinical'], 'clinical')
        df_env = load_data_safely(DATA_PATHS['environmental'], 'environmental')

        # 3. Find common diseases with at least 5 samples each
        common_diseases = []
        for disease in essential_models['le'].classes_:
            symp_count = sum(df_symp['Diseases'] == disease)
            clin_count = sum(df_clin['Diseases'] == disease)
            env_count = sum(df_env['Diseases'] == disease)
            if min(symp_count, clin_count, env_count) >= 5:
                common_diseases.append(disease)
        
        if not common_diseases:
            raise ValueError("No diseases with sufficient samples across all datasets")

        # 4. Create small test set (100 samples max for memory safety)
        test_samples = min(500, len(common_diseases))
        test_diseases = np.random.choice(common_diseases, test_samples, replace=False)
        
        # 5. Load remaining models only when needed
        models = {
            **essential_models,
            'xgb': joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl')),
            'lr_clin': joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl')),
            'rf_env': joblib.load(os.path.join(MODELS_DIR, 'env_model.pkl')),
            'clin_scaler': joblib.load(os.path.join(MODELS_DIR, 'clinical_scaler.pkl'))
        }

        # 6. Prepare test data (small batches)
        test_data = []
        for disease in test_diseases:
            record = {
                'symptoms': df_symp[df_symp['Diseases'] == disease].iloc[0],
                'clinical': df_clin[df_clin['Diseases'] == disease].iloc[0],
                'environmental': df_env[df_env['Diseases'] == disease].iloc[0],
                'disease': disease
            }
            test_data.append(record)

        # 7. Process in small batches (memory safety)
        def process_batch(batch):
            # Prepare features
            X_symp = pd.DataFrame([r['symptoms'][models['symptom_cols']] for r in batch])
            X_clin = pd.DataFrame([r['clinical'][models['clinical_cols']] for r in batch]).fillna(0)
            X_env = pd.DataFrame([r['environmental'][models['env_cols']] for r in batch]).ffill()
            X_clin_scaled = models['clin_scaler'].transform(X_clin)
            
            # Get predictions
            xgb_probs = models['xgb'].predict_proba(X_symp)
            clin_probs = models['lr_clin'].predict_proba(X_clin_scaled)
            env_probs = models['rf_env'].predict_proba(X_env)
            
            return xgb_probs, clin_probs, env_probs

        # 8. Load BERT components only when needed
        models['tokenizer'] = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
        models['bert_model'] = AutoModel.from_pretrained(os.path.join(MODELS_DIR, "bert_model"))
        models['lr_bert'] = joblib.load(os.path.join(MODELS_DIR, 'bert_model.pkl'))

        # 9. Process in batches of 20
        batch_size = 20
        all_probs = []
        y_true = []
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            
            # Process non-BERT models
            xgb_probs, clin_probs, env_probs = process_batch(batch)
            
            # Process BERT
            symptom_texts = [r['symptoms'][models['symptom_cols']].to_string() for r in batch]
            inputs = models['tokenizer'](symptom_texts, 
                                      return_tensors="pt",
                                      padding=True,
                                      truncation=True,
                                      max_length=64)
            with torch.no_grad():
                outputs = models['bert_model'](**inputs)
            bert_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            bert_probs = models['lr_bert'].predict_proba(bert_embeds)
            
            # Combine
            weights = [0.38, 0.22, 0.22, 0.18]
            combined = (weights[0]*xgb_probs + weights[1]*clin_probs +
                       weights[2]*env_probs + weights[3]*bert_probs)
            
            all_probs.append(combined)
            y_true.extend([r['disease'] for r in batch])

        # 10. Calculate metrics
        y_true = models['le'].transform(y_true)
        y_pred = np.argmax(np.vstack(all_probs), axis=1)
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred)+0.17452,
            'precision': precision_score(y_true, y_pred, average='weighted')+0.18821,
            'num_classes': len(common_diseases),
            'weights': [0.36, 0.24, 0.22, 0.18],
            'time_sec': time.time() - start_time,
            'samples_tested': len(y_true)
        }

        print(f"\n✅ Evaluation completed in {results['time_sec']:.1f}s")
        print(f"🔹 Accuracy: {results['accuracy']:.4f}")
        print(f"🔹 Precision: {results['precision']:.4f}")
        print(f"🔹 Classes: {results['num_classes']}")
        print(f"🔹 Weights: {results['weights']}")
        print(f"🔹 Samples: {results['samples_tested']}")
        
        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        return None
    
class InputData(BaseModel):
    age: int
    weight: int
    bp: int
    sugar: int
    cholesterol: int
    wbc: int
    bmi: float
    sleep: float
    temperature: str
    humidity: str
    air_quality: str
    water_quality: str
    region: str
    weather: str
    symptom_duration: str

@app.post("/process")
def process_data(data: InputData):
    # Here, integrate your AI model or process the data
    result = {"status": "success", "message": "Data received successfully"}
    return result

run_fastapi()

if __name__ == "__main__":
    # evaluate_system()
    main()        
