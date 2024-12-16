from django.http import JsonResponse
from django.shortcuts import render
from .models import *
import requests
from bs4 import BeautifulSoup
import json

def index(request):
    return render(request, 'index.html')
def analyze_page(request):
    if request.method == 'GET':
        url = request.GET.get('url')

        if not url:
            return JsonResponse({"error": "URL is required."})

        # Fetch the page content
        try:
            response = requests.get(url)
            page_content = response.text
        except Exception as e:
            return JsonResponse({"error": f"Failed to fetch the page: {str(e)}"})

        analysis_results = {}

        # 1. On-Page Optimization Analysis
        analysis_results['on_page_optimization'] = analyze_on_page_optimization(page_content)

        # 2. H1 Tag Check
        analysis_results['h1_tag'] = analyze_h1_tag(page_content)

        # 3. Schema Validation
        analysis_results['schema_validation'] = validate_schema(page_content)

        # 4. AI Content Detection (If Applicable)
        analysis_results['ai_content_detection'] = detect_ai_content(page_content)

        # 5. Page Speed Analysis
        analysis_results['page_speed'] = analyze_page_speed(url)

        # 6. Meta Tags and Keyword Suggestions
        analysis_results['meta_tags'] = check_meta_tags(page_content)

        return JsonResponse(analysis_results)

# 1. On-Page Optimization
def analyze_on_page_optimization(content):
    soup = BeautifulSoup(content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    keyword_density = get_keyword_density(content)
    
    # Example on how you might analyze on-page optimization
    return {
        'headings': [heading.text for heading in headings],
        'keyword_density': keyword_density
    }

def get_keyword_density(content):
    # Count the frequency of a specific keyword
    keyword = 'SEO'  # Example keyword
    words = content.split()
    total_words = len(words)
    keyword_count = words.count(keyword)
    density = (keyword_count / total_words) * 100
    return density

# 2. H1 Tag Check
def analyze_h1_tag(content):
    soup = BeautifulSoup(content, 'html.parser')
    h1_tag = soup.find('h1')
    title = soup.find('title')
    h1_text = h1_tag.text if h1_tag else None
    title_text = title.text if title else None

    return {
        'h1_found': bool(h1_tag),
        'h1_text': h1_text,
        'title_text': title_text,
        'title_vs_h1': h1_text == title_text  # Basic check for title-H1 consistency
    }

# 3. Schema Validation
def validate_schema(content):
    soup = BeautifulSoup(content, 'html.parser')
    schema_data = soup.find_all('script', type='application/ld+json')
    schemas = []
    for item in schema_data:
        try:
            schemas.append(json.loads(item.string))
        except json.JSONDecodeError:
            schemas.append(None)
    return schemas

# 4. AI Content Detection (Using heuristic-based approach or machine learning model)
def detect_ai_content(content):
    # Placeholder function for AI content detection.
    # Example: You could integrate an AI content detection model or heuristic-based methods here.
    # For now, returning a simple boolean.
    return 'AI content' in content  # A simple check for demonstration

# 5. Page Speed Analysis
def analyze_page_speed(url):
    try:
        # Call Lighthouse API or use Google PageSpeed API for detailed insights
        response = requests.get(f'https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}')
        data = response.json()
        # print("lighthouse result", data['lighthouseResult'])
        # print("cateogries", data['categories'])
        # print("Performance", data['performance'])
        # print("Score", data['score'])

        return data['lighthouseResult']['categories']['performance']['score']
    except Exception as e:
        return {"error": f"Failed to fetch page speed: {str(e)}"}

# 6. Meta Tags and Keyword Suggestions
def check_meta_tags(content):
    soup = BeautifulSoup(content, 'html.parser')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    
    return {
        'meta_description': meta_desc['content'] if meta_desc else None,
        'meta_keywords': meta_keywords['content'] if meta_keywords else None
    }


