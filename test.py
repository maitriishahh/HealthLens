from src.disease_predictor import predict_disease, get_symptom_list
from src.clinic_finder import find_nearby_clinics
from src.diet_plan import generate_diet_plan, save_diet_plan_pdf
from src.scraper import search_health_content
from src.credibility import evaluate_results
from src.rag import run_rag_pipeline

# Step 1 — Predict disease
symptoms = ["Itching", "Skin Rash", "Nodal Skin Eruptions"]
result = predict_disease(symptoms)
top_disease = result["ensemble_predictions"][0]["disease"]
print(f"Predicted Disease: {top_disease}")

# Step 2 — RAG summary
print("\nFetching verified information...")
web_results = search_health_content(top_disease)
evaluated = evaluate_results(web_results)
credible = [r for r in evaluated if r["credibility_label"] == "Credible"]
rag_output = run_rag_pipeline(top_disease, credible)

print(f"Total results: {len(web_results)}")
print(f"Credible results: {len(credible)}")
for r in evaluated[:5]:
    print(f"{r['credibility_label']} ({r['credibility_score']}) — {r['url']}")

print("\nSummary:")
print(rag_output["summary"])

# Step 3 — Nearby clinics
clinics = find_nearby_clinics(top_disease, "Mumbai")
print("Clinic Results:",clinics)
# print(f"\nRecommended Specialist: {clinics['specialty']}")
# print(f"Nearest Clinic: {clinics['clinics'][0]['name']}")

# Step 4 — Diet plan
diet = generate_diet_plan(top_disease, symptoms)
pdf = save_diet_plan_pdf(top_disease, diet)
print(f"\nDiet Plan PDF: {pdf}")