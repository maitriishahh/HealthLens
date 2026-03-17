import requests

def find_nearby_clinics(disease, city):
    disease_to_specialty = {
        "Fungal infection": "dermatologist",
        "Allergy": "allergist",
        "GERD": "gastroenterologist",
        "Diabetes": "endocrinologist",
        "Hypertension": "cardiologist",
        "Heart attack": "cardiologist",
        "Tuberculosis": "pulmonologist",
        "Pneumonia": "pulmonologist",
        "Bronchial Asthma": "pulmonologist",
        "Malaria": "general physician",
        "Dengue": "general physician",
        "Typhoid": "general physician",
        "Hepatitis A": "gastroenterologist",
        "Hepatitis B": "gastroenterologist",
        "Hepatitis C": "gastroenterologist",
        "Hepatitis D": "gastroenterologist",
        "Hepatitis E": "gastroenterologist",
        "Alcoholic hepatitis": "gastroenterologist",
        "Jaundice": "gastroenterologist",
        "Migraine": "neurologist",
        "Cervical spondylosis": "orthopedist",
        "Arthritis": "rheumatologist",
        "Osteoarthritis": "orthopedist",
        "Hypothyroidism": "endocrinologist",
        "Hyperthyroidism": "endocrinologist",
        "Hypoglycemia": "endocrinologist",
        "Psoriasis": "dermatologist",
        "Acne": "dermatologist",
        "Impetigo": "dermatologist",
        "Chicken pox": "general physician",
        "Common Cold": "general physician",
        "AIDS": "infectious disease specialist",
        "Urinary tract infection": "urologist",
        "Varicose veins": "vascular surgeon",
    }

    specialty = disease_to_specialty.get(disease, "general physician")

    # Step 1 — Get coordinates
    try:
        geocode_url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city, "format": "json", "limit": 1}
        headers = {"User-Agent": "HealthLensAI/1.0"}
        geo_response = requests.get(geocode_url, params=params, headers=headers, timeout=10)
        geo_data = geo_response.json()

        if not geo_data:
            return _fallback(specialty, city)

        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])

    except Exception:
        return _fallback(specialty, city)

    # Step 2 — Try Overpass API
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:10];
        (
          node["amenity"="clinic"](around:5000,{lat},{lon});
          node["amenity"="hospital"](around:5000,{lat},{lon});
          node["amenity"="doctors"](around:5000,{lat},{lon});
        );
        out body 5;
        """
        headers = {"User-Agent": "HealthLensAI/1.0"}
        overpass_response = requests.post(overpass_url, data=query, headers=headers, timeout=15)
        
        # Check if response is valid
        if overpass_response.status_code != 200 or not overpass_response.text.strip():
            return _fallback(specialty, city)

        overpass_data = overpass_response.json()
        elements = overpass_data.get("elements", [])

        if not elements:
            return _fallback(specialty, city)

        clinics = []
        for el in elements[:5]:
            tags = el.get("tags", {})
            name = tags.get("name", "Unnamed Clinic")
            address = tags.get("addr:full") or tags.get("addr:street", "Address not available")
            phone = tags.get("phone", "Not available")
            clinic_lat = el.get("lat", lat)
            clinic_lon = el.get("lon", lon)
            clinics.append({
                "name": name,
                "address": address,
                "phone": phone,
                "specialty": specialty,
                "maps_link": f"https://www.google.com/maps?q={clinic_lat},{clinic_lon}"
            })

        return {"specialty": specialty, "city": city, "clinics": clinics}

    except Exception:
        return _fallback(specialty, city)

def _fallback(specialty, city):
    """When Overpass fails — return Google Maps search link"""
    return {
        "specialty": specialty,
        "city": city,
        "clinics": [{
            "name": f"Find {specialty} near {city}",
            "address": "Click Maps link to search nearby",
            "phone": "N/A",
            "specialty": specialty,
            "maps_link": f"https://www.google.com/maps/search/{specialty}+near+{city}"
        }]
    }