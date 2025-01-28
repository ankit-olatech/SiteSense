from django.http import JsonResponse
from django.shortcuts import render
from pypsrp import FEATURES
from .models import *
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse, urljoin
import os
# from openai import OpenAI
from openai import OpenAI

client = OpenAI(api_key='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')
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from heapq import nlargest
import time
from concurrent.futures import ThreadPoolExecutor

# API KEY FOR PAGESPEED
#  AIzaSyCzMj9hqnN8lSmMIc2vMQZ2mC9N-AcNvcQ 
def index(request):
    return render(request, 'index.html')
def analyze_page(request):
    start_time = time.time()  # Record the start time

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
        print("DONE 1")

        # 2. H1 Tag Check
        analysis_results['h1_tag'] = analyze_h1_tag(page_content)
        print("DONE 2")

        # 3. Schema Validation
        analysis_results['schema_validation'] = validate_schema(page_content)
        print("DONE 3")

        # 4. AI Content Detection (If Applicable)
        analysis_results['ai_content_detection'] = detect_ai_content(page_content)
        print("DONE 4")

        # 5. Page Speed Analysis
        analysis_results['page_speed'] = analyze_page_speed(url)
        print("DONE 5: Time Consuming!")

        # 6. Meta Tags and Keyword Suggestions
        analysis_results['meta_tags'] = check_meta_tags(page_content)
        print("DONE 6")

        # ADVANCED FEATURES

         # 1. Keyword Summary
        analysis_results['keyword_summary'] = analyze_keyword_summary(page_content)
        print("DONE 7")

        # 2. Anchor Tag Suggestions
        analysis_results['anchor_tags'] = analyze_anchor_tags(page_content)
        print("DONE 8")

        # 3. URL Structure Optimization
        analysis_results['url_structure'] = analyze_url_structure(url)
        print("DONE 9")

        # 4. Robot.txt File Validation
        analysis_results['robots_txt'] = validate_robots_txt(url)
        print("DONE 10")

        # 5. XML Sitemap Validation
        analysis_results['xml_sitemap'] = validate_sitemap(url)

        print("DONE 11")
        # 6. Blog Optimization
        analysis_results['blog_optimization'] = analyze_blog_optimization(page_content, url)
        print("DONE 12")

        # 7. Broken URL Detection
        analysis_results['detect_broken_urls'] = detect_broken_urls(url)
        print("DONE 13: Time Consuming!")
        end_time = time.time()    # Record the end time
        execution_time = end_time - start_time  # Calculate the difference

        print(f"Execution time: {execution_time} seconds")



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


def validate_schema(page_content):
    """
    Validates schema.org structured data in the page content.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    scripts = soup.find_all('script', type='application/ld+json')

    schemas = []
    for script in scripts:
        try:
            data = json.loads(script.string)
            schemas.append(data)
        except json.JSONDecodeError:
            continue

    # Analyze the schema data
    results = []
    for schema in schemas:
        if isinstance(schema, dict):  # If schema is a dictionary
            schema_type = schema.get('@type', None)
            results.append({"type": schema_type, "content": schema})
        elif isinstance(schema, list):  # If schema is a list
            for item in schema:
                if isinstance(item, dict):  # Check each item in the list
                    schema_type = item.get('@type', None)
                    results.append({"type": schema_type, "content": item})
    
    return results if results else {"error": "No valid schema found"}


# Set the OpenAI API key

def detect_ai_content(content):
    # Check for AI-like patterns (heuristics)
    ai_detected = 'AI content' in content  # Simple placeholder logic for AI markers

    try:
        # Calculate readability score
        readability_score = flesch_reading_ease(content)
    except Exception as e:
        readability_score = None
        print(f"Error calculating readability score: {e}")

    try:
        # Suggest rewrite using OpenAI GPT API
        response = client.chat.completions.create(model="gpt-3.5-turbo",
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
        temperature=0.7)
        human_suggestion = response.choices[0].message.content.strip()
    except Exception as e:
        human_suggestion = f"Error generating suggestion: {str(e)}"

    return {
        "ai_detected": ai_detected or (readability_score is not None and readability_score < 50),
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



def check_meta_tags(content):
    # Parse HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Check for meta description and keywords
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})

    # Extract content text for analysis
    page_text = soup.get_text(separator=' ')

    # Analyze content for keyword suggestions
    suggested_keywords = suggest_keywords(page_text)

    # Evaluate meta keywords relevance
    existing_keywords = meta_keywords['content'].split(',') if meta_keywords else []
    irrelevant_keywords = [kw for kw in existing_keywords if kw.lower() not in page_text.lower()]

    return {
        'meta_description': meta_desc['content'] if meta_desc else None,
        'meta_keywords': existing_keywords if meta_keywords else None,
        'irrelevant_keywords': irrelevant_keywords,
        'suggested_keywords': suggested_keywords
    }


from sklearn.feature_extraction.text import TfidfVectorizer
import re

def suggest_keywords(content):
    """
    Suggest relevant keywords based on content using TF-IDF.
    """
    # Clean and preprocess content
    content = re.sub(r'[^\w\s]', '', content.lower())  # Remove punctuation and convert to lowercase

    # Define additional stop words specific to your domain
    custom_stop_words = {'that', 'your', 'there', 'where', 'what', 'can', 'use', 'like', 'will', 'one', 'make'}

    # Tokenize content into meaningful sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)

    # Combine 'english' stop words with custom stop words
    combined_stop_words = set(TfidfVectorizer(stop_words='english').get_stop_words()).union(custom_stop_words)

    # Initialize TfidfVectorizer with the updated stop words
    vectorizer = TfidfVectorizer(stop_words=list(combined_stop_words), max_features=20)

    # Calculate TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Extract keywords
    keywords = list(feature_names)

    # Further filter out irrelevant single-character words
    filtered_keywords = [kw for kw in keywords if len(kw) > 2]

    return filtered_keywords



def detect_broken_urls(url, max_threads=10):
    """
    Detect broken URLs in the given page using concurrent requests.
    
    Parameters:
        url (str): The URL of the page to analyze.
        max_threads (int): The maximum number of threads for parallel processing.
    
    Returns:
        dict: A dictionary with keys 'broken_urls' and 'valid_urls', each containing a list of URLs.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def is_broken(link):
        """Helper function to check if a URL is broken."""
        try:
            response = requests.head(link, headers=headers, allow_redirects=True, timeout=5)
            if response.status_code < 200 or response.status_code >= 300:
                return (link, response.status_code)  # Broken URL
            return (link, None)  # Valid URL
        except requests.RequestException as e:
            return (link, str(e))  # Broken URL with error message

    try:
        # Fetch the page content
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all links and resolve relative URLs
        links = soup.find_all('a', href=True)
        all_urls = list(set(urljoin(url, link['href']) for link in links))  # Deduplicate URLs

        # Filter only http(s) links
        valid_links = [link for link in all_urls if urlparse(link).scheme in ['http', 'https']]

        # Check URLs concurrently
        broken_urls = []
        valid_urls = []

        with ThreadPoolExecutor(max_threads) as executor:
            results = list(executor.map(is_broken, valid_links))
        
        for link, error in results:
            if error:
                broken_urls.append((link, error))
            else:
                valid_urls.append(link)

        return {
            "broken_urls": broken_urls,
            "valid_urls": valid_urls
        }

    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
        return {"broken_urls": [], "valid_urls": []}





# 1.1 Page SUmmary

nltk.download('punkt_tab')  # Download tokenizer for sentence splitting

# Initialize stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text by removing unwanted characters, extra spaces, and stop words.
    Args:
        text (str): The raw text to be cleaned.
    Returns:
        str: The cleaned text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a single string
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text
# If TF-IDF Fails
def fallback_keyword_extraction(cleaned_text):
    """
    Extract keywords based on word frequency as a fallback.
    Args:
        cleaned_text (str): The cleaned text.
    Returns:
        dict: Top keywords by frequency.
    """
    words = cleaned_text.split()
    word_counts = Counter(words)
    return dict(word_counts.most_common(5))

def analyze_keyword_summary(page_content):
    # Parse and clean HTML content
    soup = BeautifulSoup(page_content, 'html.parser')
    text_content = soup.get_text(separator=" ")
    cleaned_text = clean_text(text_content)
    
    if not cleaned_text or len(cleaned_text.split()) < 5:
        return {"error": "Insufficient text for meaningful analysis."}

    try:
        # Initialize TF-IDF Vectorizer with dynamic thresholds
        vectorizer = TfidfVectorizer(
            max_features=10,  # Limit to top 10 features
            stop_words=stopwords.words('english'),  # Remove common stop words
            max_df=0.85,  # Adjust upper threshold
            min_df=1  # Terms must appear at least once
        )
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])

        # Extract feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Rank keywords by score
        top_keywords = {feature_names[idx]: tfidf_scores[idx] for idx in range(len(feature_names))}

    except ValueError as e:
        # Fallback to frequency-based keywords in case of error
        keyword_freq = Counter(word_tokenize(cleaned_text))
        top_keywords = dict(keyword_freq.most_common(10))
        feature_names = list(top_keywords.keys())

    # Total word count and keyword density
    total_words = len(word_tokenize(cleaned_text))
    keyword_density = {
        kw: round((keyword_freq[kw] / total_words) * 100, 2) for kw in feature_names
    }

    # Generate summary
    sentences = sent_tokenize(text_content)
    sentence_scores = {}
    for sentence in sentences:
        for word in feature_names:
            if word in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + 1

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    summary = " ".join(summary_sentences)

    # Dynamic SEO suggestions
    seo_keywords = []
    for kw in feature_names[:5]:  # Top 5 keywords
        seo_keywords.extend([
            f"How to master {kw}",
            f"Top {kw} tools for 2025",
            f"Best {kw} strategies",
            f"Guide to {kw} success",
            f"{kw} optimization techniques"
        ])

    seo_suggestions = {
        "suggested_keywords": seo_keywords,
        "guidelines": [
            "Include these keywords in the title tag, meta description, and H1 headings.",
            "Aim for 1-2% keyword density per keyword across the content.",
            "Use long-tail variations of these keywords for better ranking."
        ]
    }

    return {
        "top_keywords": [{"keyword": kw, "score": round(score, 3)} for kw, score in top_keywords.items()],
        "keyword_density": keyword_density,
        "suggested_density": "1-2% per keyword for SEO.",
        "summary": summary,
        "gist": f"The page discusses topics centered around {', '.join(list(top_keywords.keys())[:5])}.",
        "seo_suggestions": seo_suggestions
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
nltk.download('stopwords')

def get_competitor_urls(keywords):
    
    """
    Dynamically fetch competitor blog URLs using a search query.
    """
    search_engine_url = "https://www.google.com/search"
    params = {"q": f"{keywords} blog", "num": 5}  # Search query for competitor blogs
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_engine_url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Parse search result URLs using the updated structure
        links = []
        for result in soup.select('.tF2Cxc a'):  # Adjusted selector for Google results
            href = result.get("href")
            if href:
                links.append(href)
        return links[:5]  # Return top 5 results
    except Exception as e:
        print(f"Error fetching competitor URLs: {e}")
        return []


def get_competitor_urls(keywords):
    """
    Dynamically fetch competitor blog URLs using a search query.
    """
    search_engine_url = "https://www.google.com/search"
    params = {"q": f"{keywords} blog", "num": 5}  # Search query for competitor blogs
    print(params)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_engine_url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for result in soup.select('.tF2Cxc a'):  # Adjusted selector for Google results
            href = result.get("href")
            if href:
                links.append(href)
        return links[:5]  # Return top 5 results
    except Exception as e:
        print(f"Error fetching competitor URLs: {e}")
        return []

def extract_main_keywords(text_content):
    """
    Extract main keywords from the text content using regex or basic frequency analysis.
    """
    words = re.findall(r'\b\w+\b', text_content.lower())
    stopwords = set(["the", "and", "is", "in", "to", "for", "of", "a", "on", "with", "by", "an", "as", "your", "you"])
    keyword_freq = {}

    for word in words:
        if word not in stopwords:
            keyword_freq[word] = keyword_freq.get(word, 0) + 1

    sorted_keywords = sorted(keyword_freq, key=keyword_freq.get, reverse=True)
    return " ".join(sorted_keywords[:5])


def filter_competitor_titles(competitor_titles, target_url):
    """
    Exclude competitor titles that mention the target domain or are directly related.
    """
    parsed_url = urlparse(target_url)
    target_domain = parsed_url.netloc.lower()
    target_keywords = set(re.findall(r'\w+', target_domain))  # Extract keywords from domain name

    filtered_titles = [
        title for title in competitor_titles
        if not any(keyword in title.lower() for keyword in target_keywords)  # Exclude titles mentioning the target domain
    ]
    return filtered_titles


def generate_suggested_title(current_title, competitor_titles):
    """
    Generate a keyword-rich blog title based on competitor analysis.
    """
    # Extract keywords from competitor titles
    keywords = []
    for title in competitor_titles:
        keywords.extend(re.findall(r'\b\w+\b', title.lower()))  # Extract words from competitor titles

    # Rank keywords by frequency
    keyword_freq = {}
    for word in keywords:
        keyword_freq[word] = keyword_freq.get(word, 0) + 1

    top_keywords = sorted(keyword_freq, key=keyword_freq.get, reverse=True)[:3]  # Top 3 keywords
    suggested_title = f"{current_title} - {' '.join(top_keywords)}".title()
    return suggested_title


def analyze_blog_optimization(page_content, url):
    """
    Analyze blog optimization and generate suggestions.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    title = soup.find('title').get_text(strip=True) if soup.find('title') else "No title"
    h1_tags = [h1.get_text(strip=True) for h1 in soup.find_all('h1')]

    # Extract keywords from blog content
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    keywords = meta_keywords["content"] if meta_keywords else extract_main_keywords(soup.get_text())

    # Fetch competitor blog URLs dynamically
    competitor_urls = get_competitor_urls(keywords)

    # Analyze competitor blog titles
    competitor_titles = []
    for competitor_url in competitor_urls:
        try:
            response = requests.get(competitor_url, timeout=5)
            competitor_soup = BeautifulSoup(response.text, 'html.parser')
            competitor_title = competitor_soup.find('title').get_text(strip=True) if competitor_soup.find('title') else "No title"
            competitor_titles.append(competitor_title)
        except Exception as e:
            competitor_titles.append(f"Error fetching {competitor_url}: {e}")

    # Filter competitor titles to exclude references to the target domain
    filtered_titles = filter_competitor_titles(competitor_titles, url)

    # Generate a suggested blog title
    suggested_title = generate_suggested_title(title, filtered_titles)

    return {
        "current_title": title,
        "h1_tags": h1_tags,
        "suggested_title": suggested_title,
        "competitor_titles": filtered_titles,
        "summary": "Add keyword-rich blog titles and ensure H1 tags match content."
    }


# USING GOOGLE JSON API- excluding scraping style
# from googleapiclient.discovery import build

# def get_competitor_urls_via_api(keywords, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     results = service.cse().list(q=f"{keywords} blog", cx=cse_id, num=5).execute()
#     links = [item['link'] for item in results.get('items', [])]
#     print("Competitor URLs via API:", links)
#     return links
