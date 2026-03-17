import numpy as np
import pickle
import torch
import torch.nn as nn

# ---- Load saved models ----
with open("models/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("models/symptom_list.pkl", "rb") as f:
    symptom_list = pickle.load(f)

# ---- Neural Network Definition ----
class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Load neural network
nn_model = DiseaseClassifier(len(symptom_list), 256, len(le.classes_))
nn_model.load_state_dict(torch.load("models/neural_net.pt", map_location=torch.device('cpu')))
nn_model.eval()

def get_symptom_list():
    """Returns clean symptom list for dropdown display"""
    # Clean underscores for display — stomach_pain → Stomach Pain
    return sorted([s.replace('_', ' ').title() for s in symptom_list])

def predict_disease(selected_symptoms):
    """
    Input: list of symptoms user selected from dropdown
    Output: top 3 predictions from both models + ensemble
    """
    if not selected_symptoms:
        return {"error": "Please select at least one symptom.", "predictions": []}

    # Convert selected symptoms back to model format
    selected_clean = [s.lower().replace(' ', '_') for s in selected_symptoms]

    # Build binary input vector
    input_vector = [1 if s in selected_clean else 0 for s in symptom_list]
    input_array = np.array(input_vector).reshape(1, -1)

    # XGBoost prediction
    xgb_probs = xgb_model.predict_proba(input_array)[0]
    xgb_top3_idx = np.argsort(xgb_probs)[::-1][:3]
    xgb_predictions = [
        {"disease": le.classes_[i], "confidence": round(float(xgb_probs[i]) * 100, 2)}
        for i in xgb_top3_idx
    ]

    # Neural Network prediction
    input_tensor = torch.FloatTensor(input_array)
    with torch.no_grad():
        nn_output = nn_model(input_tensor)
        nn_probs = torch.softmax(nn_output, dim=1)[0]

    nn_top3_idx = torch.argsort(nn_probs, descending=True)[:3]
    nn_predictions = [
        {"disease": le.classes_[i], "confidence": round(float(nn_probs[i]) * 100, 2)}
        for i in nn_top3_idx
    ]

    # Ensemble — average both models
    ensemble_probs = (xgb_probs + nn_probs.numpy()) / 2
    ensemble_top3_idx = np.argsort(ensemble_probs)[::-1][:3]
    ensemble_predictions = [
        {"disease": le.classes_[i], "confidence": round(float(ensemble_probs[i]) * 100, 2)}
        for i in ensemble_top3_idx
    ]

    return {
        "selected_symptoms": selected_symptoms,
        "xgboost_predictions": xgb_predictions,
        "neural_net_predictions": nn_predictions,
        "ensemble_predictions": ensemble_predictions
    }