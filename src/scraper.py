from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

CREDIBLE_DOMAINS = [
    "who.int", "pubmed.ncbi.nlm.nih.gov", "mayoclinic.org",
    "webmd.com", "nhs.uk", "healthline.com", "medlineplus.gov",
    "clevelandclinic.org", "hopkinsmedicine.org"
]

def scrape_text(url,timeout=5):
    try:
        res = requests.get(url,timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(res.text,"html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:3000]
    except:
        return ""
    
def search_health_content(condition, max_results=20):
    results = []
    ddgs = DDGS()
    for r in ddgs.text(f"{condition} symptoms treatment medical", max_results=max_results):
        url = r["href"]
        title = r["title"]
        body = r["body"]
        full_text = scrape_text(url)
        results.append({
            "url":url,
            "title":title,
            "snippet":body,
            "full_text":full_text if full_text else body
        })
    return results