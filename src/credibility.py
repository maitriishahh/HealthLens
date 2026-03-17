from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse

# ---- Trusted Domains ----
CREDIBLE_DOMAINS = [
    "who.int", "pubmed.ncbi.nlm.nih.gov", "mayoclinic.org",
    "webmd.com", "nhs.uk", "healthline.com", "medlineplus.gov",
    "clevelandclinic.org", "hopkinsmedicine.org", "apollohospitals.com",
    "kokilabenhospital.com", "ssmhealth.com", "health.harvard.edu",
    "medscape.com", "nih.gov", "cdc.gov", "icmr.gov.in",
    "wikipedia.org", "pharmeasy.in", "maxhealthcare.in",
    "lalbpathlabs.com", "manipalhospitals.com"
]

# ---- Keywords ----
CREDIBLE_KEYWORDS = [
    "symptoms", "diagnosis", "treatment", "clinical", "medical",
    "doctor", "physician", "study", "research", "evidence",
    "medication", "therapy", "syndrome", "disorder", "patient",
    "hormone", "insulin", "ovary", "prescription", "healthcare"
]

UNRELIABLE_KEYWORDS = [
    "miracle", "cure overnight", "guaranteed", "secret remedy",
    "doctors hate", "one weird trick", "instant cure",
    "100% natural", "detox", "superfood", "magic"
]

# ---- Reference Medical Text ----
# This is what "credible medical content" sounds like
# SentenceTransformer will compare each scraped text AGAINST this
MEDICAL_REFERENCE = """
Medical diagnosis and clinical treatment requires evidence-based approaches recommended by 
qualified physicians and healthcare professionals. Symptoms should be evaluated through 
proper clinical examination, laboratory tests, and imaging studies. Peer-reviewed research 
and medical journals provide treatment guidelines for patients. Medications, therapies and 
lifestyle modifications should be prescribed based on scientific evidence. Healthcare 
providers follow established protocols for disease prevention, diagnosis and patient care. 
Medical conditions require proper understanding of pathophysiology, risk factors, 
complications and long-term management strategies supported by clinical studies.
"""

# ---- Load SentenceTransformer ----
# all-MiniLM-L6-v2 is small (~90MB), fast, and great for semantic similarity
# You already used SentenceTransformers in InsightNotes — same library!
print("Loading SentenceTransformer model...")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
reference_embedding = semantic_model.encode(MEDICAL_REFERENCE, convert_to_tensor=True)
print("Model loaded!")

# ---- Helper Functions ----

def is_credible_domain(url):
    """Check if URL belongs to a trusted medical domain"""
    try:
        domain = urlparse(url).netloc
        domain = domain.replace("www.", "")
        return any(trusted in domain for trusted in CREDIBLE_DOMAINS)
    except:
        return False

def get_semantic_score(text):
    """
    DL layer — SentenceTransformer computes how semantically similar
    the scraped text is to our reference medical paragraph
    Returns cosine similarity score between 0 and 1
    """
    if not text or len(text) < 50:
        return 0.0

    # Encode scraped text into embedding vector
    text_embedding = semantic_model.encode(text[:500], convert_to_tensor=True)

    # Compare with reference medical text using cosine similarity
    # cosine similarity = 1 means identical meaning, 0 means completely different
    similarity = util.cos_sim(text_embedding, reference_embedding)
    return round(float(similarity), 2)

def get_keyword_score(text):
    """
    Rule-based layer — keyword frequency scoring
    Adds explainability on top of DL score
    """
    text_lower = text.lower()
    credible_hits = sum(1 for w in CREDIBLE_KEYWORDS if w in text_lower)
    unreliable_hits = sum(1 for w in UNRELIABLE_KEYWORDS if w in text_lower)
    boost = min(0.2, credible_hits * 0.01) - (unreliable_hits * 0.05)
    return round(boost, 2)

def score_credibility(text, url):
    """
    Final score = DL semantic similarity + domain trust + keyword boost
    
    Three layers:
    1. SentenceTransformer semantic score (main DL signal)
    2. Domain whitelist boost (trust signal)
    3. Keyword score (explainability layer)
    """
    semantic_score = get_semantic_score(text)
    domain_boost = 0.2 if is_credible_domain(url) else 0.0
    keyword_boost = get_keyword_score(text)

    final_score = semantic_score + domain_boost + keyword_boost
    final_score = max(0.0, min(1.0, final_score))  # clamp 0 to 1
    return round(final_score, 2)

def evaluate_results(results):
    """
    Takes scraper results → scores each → sorts by credibility
    """
    evaluated = []

    for r in results:
        final_score = score_credibility(r["full_text"], r["url"])

        if final_score >= 0.25:
            label = "Credible"
        elif final_score >= 0.15:
            label = "Moderate"
        else:
            label = "Unreliable"

        evaluated.append({
            **r,
            "credibility_score": final_score,
            "credibility_label": label
        })

    evaluated.sort(key=lambda x: x["credibility_score"], reverse=True)
    return evaluated