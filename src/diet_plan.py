from groq import Groq
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph, Spacer
from reportlab.lib.units import inch

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_diet_plan(disease, symptoms):
    """ Generates evidence-based diet plan for predicted disease
    Grounded in medical nutrition guidelines - not random ai advice
    """

    prompt =  f"""You are a clinical nutritionist assistant.
Generate a structured 7-day diet plan for a patient diagnosed with {disease}.
Patient symptoms: {', '.join(symptoms)}

Follow these rules strictly:
1. Base recommendations on established medical nutrition guidelines only
2. Include: Foods to eat, Foods to avoid, Sample daily meal plan
3. Keep language simple and practical
4. End with: "Please consult a registered dietitian for personalized advice."
5. Keep response under 400 words

Format exactly like this:
DIET PLAN FOR: {disease}

FOODS TO EAT:
- [food 1]
- [food 2]

FOODS TO AVOID:
- [food 1]
- [food 2]

DAILY MEAL PLAN:
Breakfast: [meal]
Mid-morning: [snack]
Lunch: [meal]
Evening: [snack]
Dinner: [meal]

IMPORTANT NOTE:
Please consult a registered dietitian for personalized advice.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

def save_diet_plan_pdf(disease, diet_text):
    """
    Saves diet plan as PDF in outputs/ folder
    """
    os.makedirs("outputs", exist_ok=True)
    filename = f"outputs/diet_plan_{disease.replace(' ', '_')}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20
    )
    story.append(Paragraph(f"HealthLens AI — Diet Plan", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Content — split by lines
    for line in diet_text.split('\n'):
        if line.strip():
            if line.isupper():
                # Section headers
                story.append(Paragraph(line, styles['Heading2']))
            else:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return filename