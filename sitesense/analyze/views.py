from django.http import JsonResponse
from django.shortcuts import render
from pypsrp import FEATURES
from .models import *
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
import re
import os
import openai
from textstat import flesch_reading_ease

# client = OpenAI(api_key='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')
# API KEY FOR PAGESPEED
#  AIzaSyCzMj9hqnN8lSmMIc2vMQZ2mC9N-AcNvcQ 

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

        # ADVANCED FEATURES

         # 1. Keyword Summary
        analysis_results['keyword_summary'] = analyze_keyword_summary(page_content)

        # 2. Anchor Tag Suggestions
        analysis_results['anchor_tags'] = analyze_anchor_tags(page_content)

        # 3. URL Structure Optimization
        analysis_results['url_structure'] = analyze_url_structure(url)

        # 4. Robot.txt File Validation
        analysis_results['robots_txt'] = validate_robots_txt(url)

        # 5. XML Sitemap Validation
        analysis_results['xml_sitemap'] = validate_sitemap(url)

        # 6. Blog Optimization
        analysis_results['blog_optimization'] = analyze_blog_optimization(page_content, url)


        return JsonResponse(analysis_results)

# 1. On-Page Optimization
def analyze_on_page_optimization(content):
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find all headings (h1 to h6)
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # Organize headings by their tag (h1, h2, etc.)
    heading_hierarchy = {
        'h1': [],
        'h2': [],
        'h3': [],
        'h4': [],
        'h5': [],
        'h6': []
    }
    
    for heading in headings:
        tag = heading.name  # This gets the name of the tag (e.g., 'h1', 'h2', etc.)
        heading_hierarchy[tag].append(heading.text.strip())  # Append the heading text
    
    # Calculate keyword density (Placeholder for actual function)
    keyword_density = get_keyword_density(content)

    return {
        'headings': heading_hierarchy,  # Now includes each heading level and its corresponding text
        'keyword_density': keyword_density
    }


def get_keyword_density(content):
    # Count the frequency of a specific keyword
    keyword = 'SEO'  # Example keyword, change it to by seeing SEO friendly words
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
    recommendations = []

    for item in schema_data:
        try:
            schema = json.loads(item.string)
            schemas.append(schema)

            # Check for common schema types
            schema_type = schema.get('@type', None)
            if schema_type:
                # Suggestions for improvement
                if schema_type == "Product" and "offers" not in schema:
                    recommendations.append("Add 'offers' property to the Product schema for pricing details.")
                if schema_type == "Article" and "author" not in schema:
                    recommendations.append("Add 'author' property to the Article schema for attribution.")
                if schema_type == "Review" and "reviewRating" not in schema:
                    recommendations.append("Include 'reviewRating' in Review schema for better user insights.")
            else:
                recommendations.append("Schema type is missing. Add '@type' to define the schema's purpose.")

        except json.JSONDecodeError:
            recommendations.append("Invalid JSON-LD detected. Fix formatting issues.")

    # Detect Missing Schema Types
    if not schema_data:
        recommendations.append("No schema markup detected. Add appropriate schema types like Product, Article, or Review based on page content.")
    else:
        if not any(schema.get('@type') == "BreadcrumbList" for schema in schemas):
            recommendations.append("Add 'BreadcrumbList' schema to improve navigation.")
        if not any(schema.get('@type') == "Organization" for schema in schemas):
            recommendations.append("Add 'Organization' schema to provide details about the website owner.")

    return {
        "schemas_detected": schemas,
        "recommendations": recommendations
    }



def detect_ai_content(content):
    # Check for AI-like patterns (heuristics)
    ai_detected = 'AI content' in content  # Simple placeholder logic for AI markers
    readability_score = flesch_reading_ease(content)

    try:
        # Suggest rewrite using OpenAI GPT API
        openai.api_key = 'k-proj-uu94BHcVuwNsiLbV9GUssPFWrSecK7EeyvFK17IcPINslOJu-OIbrt1nCknmcgXrWQH30LzbNiT3BlbkFJot0E3juP913Aaw7T4HUC9puEqD2nBEgBYwp0IjNeaexN2WV7yYlsGydF_Jm-uFoFNHq_uOUVoA'
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that rewrites text to sound more human."
                },
                {
                    "role": "user",
                    "content": f"The following text may be AI-generated. Rewrite it to sound more human: {content}"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        human_suggestion = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        human_suggestion = f"Error generating suggestion: {str(e)}"

    return {
        "ai_detected": ai_detected or readability_score < 50,
        "readability_score": readability_score,
        "human_suggestion": human_suggestion
    }

# 5. Page Speed Analysis
def analyze_page_speed(url):
    try:
        # Call Google PageSpeed API
        response = requests.get(
            f'https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key=AIzaSyCzMj9hqnN8lSmMIc2vMQZ2mC9N-AcNvcQ'
        )
        data = response.json()

        # Extract detailed insights from the response
        lighthouse_result = data.get('lighthouseResult', {})
        categories = lighthouse_result.get('categories', {})
        audits = lighthouse_result.get('audits', {})

        # Organize results into a dictionary
        page_speed_results = {
            "Performance_Score": categories.get('performance', {}).get('score', 'N/A'),
            "First_Contentful_Paint": audits.get('first-contentful-paint', {}).get('displayValue', 'N/A'),
            "Speed_Index": audits.get('speed-index', {}).get('displayValue', 'N/A'),
            "Largest_Contentful_Paint": audits.get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
            "Cumulative_Layout_Shift": audits.get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
            "Time_to_Interactive": audits.get('interactive', {}).get('displayValue', 'N/A'),
            "Total_Blocking_Time": audits.get('total-blocking-time', {}).get('displayValue', 'N/A')
        }

        return page_speed_results

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





# ADVANCVED FEATURES
# ---------------------------------
# 1. Keyword Summary
# ---------------------------------
def analyze_keyword_summary(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    text_content = soup.get_text()

    # Count words and extract keywords
    words = re.findall(r'\w+', text_content.lower())
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Ignore short words
            word_freq[word] = word_freq.get(word, 0) + 1

    total_words = len(words)
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10 keywords

    # Keyword density analysis
    keyword_density = {kw: round((count / total_words) * 100, 2) for kw, count in sorted_keywords}

    return {
        "top_keywords": sorted_keywords,
        "keyword_density": keyword_density,
        "suggested_density": "1-2% per keyword for SEO."
    }


# ---------------------------------
# 2. Anchor Tag Suggestions
# ---------------------------------
def analyze_anchor_tags(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    anchor_tags = soup.find_all('a')

    missing_anchors = []
    for anchor in anchor_tags:
        text = anchor.get_text(strip=True)
        if not text or text == "click here":
            missing_anchors.append(str(anchor))

    return {
        "total_anchors": len(anchor_tags),
        "missing_anchors": len(missing_anchors),
        "suggestion": "Replace generic anchor texts with keyword-rich phrases."
    }


# ---------------------------------
# 3. URL Structure Optimization
# ---------------------------------
def analyze_url_structure(url):
    parsed_url = urlparse(url)
    path = parsed_url.path

    if len(path) > 60 or not path.endswith('/'):
        return {
            "url": url,
            "issue": "URL is too long or improperly formatted.",
            "suggestion": "Keep URLs under 60 characters, use hyphens, and avoid trailing symbols."
        }
    return {"url": url, "status": "URL is optimized."}


# ---------------------------------
# 4. Robot.txt File Validation
# ---------------------------------
def validate_robots_txt(url):
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    robots_url = f"{base_url}/robots.txt"

    try:
        response = requests.get(robots_url)
        if response.status_code == 200:
            content = response.text
            return {"status": "Exists", "content": content}
        else:
            return {"status": "Missing", "suggestion": "Add a robots.txt file to control crawling."}
    except Exception as e:
        return {"error": f"Failed to check robots.txt: {str(e)}"}


# ---------------------------------
# 5. XML Sitemap Validation
# ---------------------------------
def validate_sitemap(url):
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    sitemap_url = f"{base_url}/sitemap.xml"

    try:
        response = requests.get(sitemap_url)
        if response.status_code == 200:
            return {"status": "Exists", "sitemap_url": sitemap_url}
        else:
            return {
                "status": "Missing",
                "suggestion": "Create a sitemap.xml file for better crawling."
            }
    except Exception as e:
        return {"error": f"Failed to check sitemap: {str(e)}"}


# ---------------------------------
# 6. Blog Optimization
# ---------------------------------
def analyze_blog_optimization(page_content, url):
    soup = BeautifulSoup(page_content, 'html.parser')

    # Detect blog-like structures
    title = soup.find('title').get_text(strip=True) if soup.find('title') else "No title"
    h1_tags = [h1.get_text(strip=True) for h1 in soup.find_all('h1')]

    # Generate suggested blog title
    suggested_title = f"Ultimate Guide to {title.split()[0]} | Tips & Insights"

    return {
        "current_title": title,
        "h1_tags": h1_tags,
        "suggested_title": suggested_title,
        "summary": "Add keyword-rich blog titles and ensure H1 tags match content."
    }