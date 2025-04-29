from django.http import JsonResponse
from django.shortcuts import render
# --- Removed OpenAI ---
# from openai import OpenAI
from rake_nltk import Rake
# from pypsrp import FEATURES # Assuming this is not used elsewhere
# from .models import * # Assuming models are not directly used in this snippet
import requests
from bs4 import BeautifulSoup, Comment # Added Comment
import json
from urllib.parse import urlparse, urljoin
import os
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Added Lemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from heapq import nlargest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import string

# --- Removed Transformers ---
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- Removed googlesearch (Used in blog optimization, keep if that feature is essential) ---
# from googlesearch import search


# Ensure NLTK data is available
try:

    nltk.data.find('corpora/wordnet')

except LookupError:

    nltk.download('wordnet')


try:

    nltk.data.find('tokenizers/punkt')

except LookupError:

    nltk.download('punkt')


try:

    nltk.data.find('corpora/stopwords')

except LookupError:

    nltk.download('stopwords')

# --- Constants ---
# API KEY FOR PAGESPEED (Keep as is)
PAGESPEED_API_KEY = 'AIzaSyCzMj9hqnN8lSmMIc2vMQZ2mC9N-AcNvcQ' # Use a constant

# Extended Stopwords
CUSTOM_STOP_WORDS = {
    'http', 'https', 'www', 'com', 'org', 'net', 'gov', 'edu', 'like', 'get', 'use',
    'also', 'said', 'could', 'would', 'should', 'make', 'one', 'two', 'see', 'may',
    'might', 'well', 'good', 'great', 'page', 'site', 'website', 'click', 'here',
    'learn', 'more', 'find', 'out', 'read', 'content', 'information', 'visit',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'privacy', 'policy', 'terms',
    'service', 'copyright', 'rights', 'reserved', 'inc', 'llc'
}
ENGLISH_STOP_WORDS = set(stopwords.words('english')).union(CUSTOM_STOP_WORDS)

# --- Helper Functions ---

def get_visible_text(soup):
    """Extracts visible text content from the main part of the page."""
    # Attempt to find the main content area
    main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main')

    if not main_content:
        main_content = soup.body # Fallback to body if specific main tags aren't found

    if not main_content:
        return "" # No body or main content found

    # Remove common non-content tags
    tags_to_remove = ['script', 'style', 'nav', 'footer', 'aside', 'header', 'form', 'button', 'select', 'option']
    for tag in main_content.find_all(tags_to_remove):
        tag.decompose()

    # Remove comments
    comments = main_content.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Get text, separating paragraphs and list items for better structure
    lines = []
    for element in main_content.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th', 'div']):
        text = element.get_text(separator=' ', strip=True)
        if text:
            lines.append(text)

    full_text = ' '.join(lines)
    # Further clean extraneous whitespace
    full_text = re.sub(r'\s{2,}', ' ', full_text).strip()
    return full_text


def clean_text_for_keywords(text):
    """Cleans text specifically for keyword analysis (lowercase, remove punct/nums, lemmatize, stopwords)."""
    if not text:
        return "", []

    lemmatizer = WordNetLemmatizer()

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    words = word_tokenize(text)

    # Lemmatize and remove stopwords
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in ENGLISH_STOP_WORDS and len(word) > 2 # Keep words longer than 2 chars
    ]

    cleaned_text_string = ' '.join(filtered_words)
    return cleaned_text_string, filtered_words


def extract_keywords_combined_v2(original_text, cleaned_text, cleaned_words, top_n=20):
    """
    Extracts keywords using TF-IDF and RAKE, focusing on cleaned text for TF-IDF
    and original structure for RAKE.
    """
    if not cleaned_words:
        return {}, 0

    keywords = {}
    total_word_count = len(cleaned_words) # Count based on cleaned list

    # --- Method 1: TF-IDF on Cleaned Text ---
    try:
        # Use cleaned text string for TF-IDF
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), # Uni- and Bi-grams
            max_features=100,   # Limit feature space
            stop_words=list(ENGLISH_STOP_WORDS) # Ensure stopwords are handled
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        tfidf_scores = {feature_names[i]: scores[i] for i in range(len(feature_names))}
    except ValueError: # Handle case with very short text
        tfidf_scores = {}

    # --- Method 2: RAKE on Original Text (for phrase structure) ---
    try:
        r = Rake(stopwords=ENGLISH_STOP_WORDS, min_length=2, max_length=3) # Focus on 2-3 word phrases
        # Use original text to preserve sentence structure for RAKE
        r.extract_keywords_from_text(original_text)
        # Get phrase scores, filter out single-word results if TF-IDF handles them
        rake_scores_phrases = {phrase: score for phrase, score in r.get_ranked_phrases_with_scores() if len(phrase.split()) > 1}
        # Optional: Get word scores from RAKE if needed, but TF-IDF is often better for single words
        # rake_scores_words = r.get_word_degrees()
    except Exception:
        rake_scores_phrases = {}


    # --- Combine Scores (Prioritize TF-IDF for relevance, RAKE for phrases) ---
    combined_scores = {}

    # Add TF-IDF scores (weighted higher for single/bigram relevance)
    for word, score in tfidf_scores.items():
        combined_scores[word] = combined_scores.get(word, 0) + score * 0.6

    # Add RAKE phrase scores (weighted for multi-word importance)
    for phrase, score in rake_scores_phrases.items():
         # Simple check: only add if it seems substantial enough
        if score > 1.5: # Adjust this threshold based on observation
            combined_scores[phrase] = combined_scores.get(phrase, 0) + score * 0.4


    # Get top N keywords/phrases
    top_keywords_list = nlargest(top_n, combined_scores.items(), key=lambda item: item[1])

    # Calculate density based on cleaned words list
    final_keywords = {}
    word_counts = Counter(cleaned_words)
    for kw, score in top_keywords_list:
        # Approximate count for phrases by checking constituent words
        phrase_words = kw.split()
        count = 0
        if len(phrase_words) > 1:
            # Heuristic: count occurrences where first two words appear near each other
            # This is complex to do accurately without full text indexing.
            # Simpler approach: average counts of constituent words or count first word.
            # Using count of the first word as a proxy:
            first_word = phrase_words[0]
            lemmatized_first = WordNetLemmatizer().lemmatize(first_word.lower())
            count = word_counts.get(lemmatized_first, 0)

            # Alternative: Count exact phrase matches (less robust to variations)
            # count = original_text.lower().count(kw) # Count in original text
        else:
            # Single word: use lemmatized count
             lemmatized_kw = WordNetLemmatizer().lemmatize(kw.lower())
             count = word_counts.get(lemmatized_kw, 0)

        density = (count / total_word_count) * 100 if total_word_count > 0 else 0
        final_keywords[kw] = {"score": round(score, 4), "count": count, "density": round(density, 2)}


    return final_keywords, total_word_count


def generate_seo_suggestions_v2(keywords_data):
    """
    Generates specific SEO suggestions based on extracted keyword data.
    """
    if not keywords_data:
        return {
            "strategic_focus": "No significant keywords found. Ensure the page has sufficient, relevant text content.",
            "on_page_tips": [],
            "content_ideas": [],
            "technical_seo": []
        }

    # Sort keywords by score to identify primary/secondary
    sorted_keywords = sorted(keywords_data.items(), key=lambda item: item[1]['score'], reverse=True)
    primary_keywords = [kw[0] for kw in sorted_keywords[:3]] # Top 3 as primary
    secondary_keywords = [kw[0] for kw in sorted_keywords[3:8]] # Next 5 as secondary

    suggestions = {
        "strategic_focus": f"Focus on '{primary_keywords[0]}' as the primary keyword. Support with '{', '.join(primary_keywords[1:])}' and secondary terms like '{', '.join(secondary_keywords)}'.",
        "on_page_tips": [],
        "content_ideas": [],
        "technical_seo": [
            "Ensure your XML sitemap is submitted to search engines.",
            "Check for mobile-friendliness using Google's Mobile-Friendly Test.",
            "Review page load speed insights and address critical issues."
        ]
    }

    # Title and Headings
    suggestions["on_page_tips"].append(f"**Title Tag:** Include '{primary_keywords[0]}' ideally near the beginning.")
    suggestions["on_page_tips"].append(f"**H1 Tag:** Ensure the main H1 heading clearly reflects the page topic and contains '{primary_keywords[0]}' or a close variant.")
    suggestions["on_page_tips"].append(f"**Subheadings (H2, H3):** Use variations of primary keywords (e.g., '{primary_keywords[1]}') and secondary keywords ('{secondary_keywords[0]}', '{secondary_keywords[1]}') in H2s and H3s to structure content and improve readability.")

    # Body Content
    suggestions["on_page_tips"].append(f"**Introduction:** Mention '{primary_keywords[0]}' within the first 100 words.")
    suggestions["on_page_tips"].append(f"**Keyword Density:** Aim for a natural density. The calculated densities ({ {kw: data['density'] for kw, data in sorted_keywords[:5]} }) seem reasonable/low/high [manual interpretation needed here based on values]. Aim for 1-2% for primary terms, used naturally.")
    suggestions["on_page_tips"].append("**Semantic Keywords:** Include related terms and synonyms for your main keywords. For example, if discussing 'SEO analysis', mention 'keyword research', 'on-page optimization', 'technical SEO', 'backlink analysis'.")
    suggestions["on_page_tips"].append("**Readability:** Ensure content is easy to read (check Flesch score). Break up long paragraphs and use bullet points.")

    # Images and Links
    suggestions["on_page_tips"].append(f"**Image Alt Text:** Use descriptive alt text for images, incorporating relevant keywords like '{primary_keywords[1]}' where appropriate.")
    suggestions["on_page_tips"].append(f"**Internal Linking:** Link relevant keywords (e.g., '{secondary_keywords[0]}') to other relevant pages on your site using descriptive anchor text.")
    # suggestions["on_page_tips"].append(f"**External Linking:** Link out to authoritative sources where relevant, but ensure links open in new tabs if needed.") # Optional

    # Content Ideas
    suggestions["content_ideas"].append(f"Create a detailed guide or FAQ section answering common questions about '{primary_keywords[0]}'.")
    suggestions["content_ideas"].append(f"Expand on sub-topics related to '{primary_keywords[1]}' or '{secondary_keywords[0]}'.")
    suggestions["content_ideas"].append(f"Consider adding case studies, examples, or tutorials related to '{primary_keywords[0]}'.")
    suggestions["content_ideas"].append(f"Develop content around long-tail variations like 'how to use {primary_keywords[0]} for [specific goal]' or 'best {primary_keywords[1]} tools'.")


    # Meta Description Suggestion
    suggestions["technical_seo"].append(f"**Meta Description:** Craft a compelling meta description (under 160 chars) that includes '{primary_keywords[0]}' and encourages clicks. Example: 'Learn effective {primary_keywords[0]} techniques to boost your rankings. Discover insights on {primary_keywords[1]} and {secondary_keywords[0]}. Read more!'")


    return suggestions


# --- Main Analysis Functions (Modified) ---

def analyze_page(request):
    if request.method == 'GET':
        url = request.GET.get('url')
        if not url:
            return JsonResponse({"error": "URL is required."})

        try:
            headers = {'User-Agent': 'Mozilla/5.0'} # Simple user agent
            response = requests.get(url, headers=headers, timeout=15) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            page_content = response.text
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                 return JsonResponse({"error": f"Invalid content type: {content_type}. Expected HTML."})

        except requests.exceptions.RequestException as e:
            return JsonResponse({"error": f"Failed to fetch the page: {str(e)}"})
        except Exception as e:
             return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"})


        analysis_results = {}
        start_time = time.time()

        # --- Run analyses concurrently ---
        with ThreadPoolExecutor(max_workers=10) as executor: # Limit workers
            # Submit tasks that don't depend on each other first
            future_to_analysis = {
                # Independent tasks
                executor.submit(analyze_h1_tag, page_content): 'h1_tag',
                executor.submit(validate_schema, page_content): 'schema_validation',
                executor.submit(analyze_page_speed, url): 'page_speed', # Uses URL
                executor.submit(analyze_anchor_tags, page_content): 'anchor_tags',
                executor.submit(analyze_url_structure, url): 'url_structure', # Uses URL
                executor.submit(validate_robots_txt, url): 'robots_txt', # Uses URL
                executor.submit(validate_sitemap, url): 'xml_sitemap', # Uses URL
                executor.submit(detect_broken_urls, url, page_content): 'detect_broken_urls', # Pass content to avoid re-fetching
                executor.submit(detect_ai_content_v2, page_content): 'ai_content_detection', # Updated AI detect
                executor.submit(check_meta_tags_v2, page_content): 'meta_tags', # Updated meta check
                executor.submit(analyze_on_page_optimization_v2, page_content): 'on_page_optimization', # Updated on-page
                 # Keyword analysis depends on content processing
                executor.submit(analyze_keyword_summary_v2, page_content): 'keyword_summary', # NEW: Central keyword analysis
                # Blog optimization might depend on keywords, run it if needed
                # executor.submit(analyze_blog_optimization, page_content, url): 'blog_optimization',
            }

            for future in as_completed(future_to_analysis):
                analysis_name = future_to_analysis[future]
                try:
                    analysis_results[analysis_name] = future.result()
                except Exception as e:
                    print(f"Error during analysis '{analysis_name}': {e}") # Log error server-side
                    analysis_results[analysis_name] = {"error": f"Analysis failed: {str(e)}"}

        end_time = time.time()
        print(f"Analysis for {url} completed in {end_time - start_time:.2f} seconds")

        return JsonResponse(analysis_results)

    # If not GET request
    return JsonResponse({"error": "Invalid request method."}, status=405)


# 1. On-Page Optimization (Simplified - Keyword density moved)
def analyze_on_page_optimization_v2(content):
    """Analyzes heading structure."""
    soup = BeautifulSoup(content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    heading_hierarchy = {f'h{i}': [] for i in range(1, 7)}
    issues = []

    if not soup.find('h1'):
        issues.append("Missing H1 tag. Every page should have exactly one H1.")
    elif len(soup.find_all('h1')) > 1:
         issues.append("Multiple H1 tags found. Best practice is to use only one H1 per page.")

    current_level = 0
    for heading in headings:
        try:
            tag = heading.name
            level = int(tag[1])
            text = heading.text.strip()
            if not text:
                 issues.append(f"Empty {tag.upper()} tag found.")
            else:
                heading_hierarchy[tag].append(text)

            # Basic hierarchy check (e.g., H3 should follow H2 or H3, not H1 directly)
            if level > current_level + 1:
                 issues.append(f"Heading hierarchy jump: Found <{tag}> potentially without a preceding <h{level-1}>.")
            current_level = level

        except (ValueError, IndexError):
            continue # Should not happen with the find_all filter

    return {
        'headings_by_level': heading_hierarchy,
        'heading_issues': issues,
        'recommendation': "Ensure headings form a logical structure (H1 > H2 > H3...) and are not skipped. Use headings to outline page content clearly."
        # Keyword density is now handled in analyze_keyword_summary_v2
    }

# Keyword Density function (removed - integrated into keyword summary)
# def get_keyword_density(content): ...

# 2. H1 Tag Check (Remains similar, could be integrated into On-Page)
def analyze_h1_tag(content):
    soup = BeautifulSoup(content, 'html.parser')
    h1_tags = soup.find_all('h1')
    title_tag = soup.find('title')

    h1_texts = [h1.text.strip() for h1 in h1_tags if h1.text.strip()]
    title_text = title_tag.text.strip() if title_tag else None
    num_h1 = len(h1_texts)
    h1_found = num_h1 > 0
    h1_text = h1_texts[0] if num_h1 == 1 else None # Report text only if exactly one non-empty H1

    status = "OK"
    message = f"Found {num_h1} non-empty H1 tag(s)."
    if num_h1 == 0:
        status = "Error"
        message = "No H1 tag found. Crucial for SEO and accessibility."
    elif num_h1 > 1:
        status = "Warning"
        message = f"Found {num_h1} non-empty H1 tags. Best practice is one H1 per page."

    title_match = False
    if h1_text and title_text and h1_text.lower() == title_text.lower():
         title_match = True


    return {
        'status': status,
        'message': message,
        'h1_found': h1_found,
        'h1_count': num_h1,
        'h1_texts': h1_texts, # List all found H1 texts
        'title_text': title_text,
        'title_h1_match': title_match,
        'recommendation': "Ensure exactly one, descriptive H1 tag per page, closely related to the page's title and main topic."
    }

# 3. Schema Validation (No changes needed based on request)
def validate_schema(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    scripts = soup.find_all('script', type='application/ld+json')
    schemas = []
    errors = []

    for script in scripts:
        try:
            # Handle potential comments within script tags
            script_content = script.string
            if script_content:
                 # Basic cleaning of potential JS comments that might invalidate JSON
                 script_content = re.sub(r'//.*?\n|/\*.*?\*/', '', script_content, flags=re.S)
                 data = json.loads(script_content)
                 schemas.append(data)
            else:
                 errors.append("Found empty <script type='application/ld+json'> tag.")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON-LD found: {e}. Content snippet: {script.string[:100]}...")
        except Exception as e:
            errors.append(f"Error processing script tag: {e}")


    results = []
    if schemas:
        for schema in schemas:
            if isinstance(schema, dict):
                schema_type = schema.get('@type', 'Type not specified')
                results.append({"type": schema_type, "content": schema})
            elif isinstance(schema, list):
                for item in schema:
                    if isinstance(item, dict):
                        schema_type = item.get('@type', 'Type not specified')
                        results.append({"type": schema_type, "content": item})
    elif not errors:
         errors.append("No schema.org JSON-LD structured data found.")


    return {
        "found_schemas": results,
        "validation_errors": errors,
        "recommendation": "Use schema.org structured data (JSON-LD recommended) to help search engines understand your content context (e.g., Article, Product, Event)."
        }


# 4. AI Content Detection (V2 - No OpenAI/T5)
def detect_ai_content_v2(content):
    """
    Detects potential AI content using heuristics and readability, without external AI models.
    """
    soup = BeautifulSoup(content, 'html.parser')
    page_text = get_visible_text(soup) # Use the function to get main content text

    if len(page_text.strip()) < 100: # Need sufficient text
        return {
            "potential_ai_flag": False,
            "readability_score": None,
            "message": "Not enough text content to perform meaningful analysis.",
            "recommendation": "Ensure pages have substantial, unique content."
        }

    ai_detected = False
    readability_score = None
    issues = []

    # Heuristic 1: Check for common AI generation footprints (simple check)
    ai_footprints = ["generated by ai", "language model", "openai", "gpt-3", "gpt-4", "claude", "bard", "llm"]
    page_text_lower = page_text.lower()
    for footprint in ai_footprints:
        if footprint in page_text_lower:
            ai_detected = True
            issues.append(f"Found potential AI footprint phrase: '{footprint}'.")
            break # Stop after first finding

    # Heuristic 2: Check readability score
    try:
        # Calculate score on the extracted visible text
        readability_score = flesch_reading_ease(page_text)
        if readability_score < 30: # Very difficult - can sometimes be AI jargon
             # ai_detected = True # Commented out: Low score isn't definitively AI
             issues.append(f"Readability score is very low ({readability_score:.2f}), indicating complex language. Review for clarity.")
        elif readability_score > 90: # Very easy - can sometimes be overly simplistic AI output
             # ai_detected = True # Commented out: High score isn't definitively AI
             issues.append(f"Readability score is very high ({readability_score:.2f}), indicating very simple language. Ensure sufficient depth.")
    except Exception as e:
        issues.append(f"Could not calculate readability score: {e}")

    # Heuristic 3: Repetitiveness (Basic check)
    # Tokenize and check for overly repeated phrases (e.g., 3-grams)
    try:
        words = word_tokenize(page_text_lower)
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if trigrams:
             trigram_counts = Counter(trigrams)
             most_common = trigram_counts.most_common(1)[0]
             # If a trigram appears many times relative to text length
             if most_common[1] > 5 and most_common[1] > len(words) / 100:
                  ai_detected = True
                  issues.append(f"High repetition of phrase '{most_common[0]}' ({most_common[1]} times). Could indicate unnatural writing.")
    except Exception as e:
        issues.append(f"Could not perform repetition analysis: {e}")


    # Combine results
    message = "AI content heuristics checked."
    if ai_detected:
        message = "Potential indicators of AI-generated or unnatural content found."

    return {
        "potential_ai_flag": ai_detected,
        "readability_score": round(readability_score, 2) if readability_score is not None else None,
        "analysis_issues": issues,
        "message": message,
        "recommendation": "Review content for originality, natural language flow, accuracy, and unique insights. While heuristics can flag patterns, human review is essential. Focus on providing genuine value to the reader."
    }


# 5. Page Speed Analysis (No changes needed)
def analyze_page_speed(url):
    try:
        api_url = f'https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={PAGESPEED_API_KEY}&category=PERFORMANCE&strategy=DESKTOP' # Added category/strategy
        desktop_response = requests.get(api_url, timeout=60) # Increase timeout for API call
        desktop_response.raise_for_status()
        desktop_data = desktop_response.json()

        api_url_mobile = f'https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={PAGESPEED_API_KEY}&category=PERFORMANCE&strategy=MOBILE'
        mobile_response = requests.get(api_url_mobile, timeout=60)
        mobile_response.raise_for_status()
        mobile_data = mobile_response.json()

        def extract_metrics(data):
            lighthouse_result = data.get('lighthouseResult', {})
            categories = lighthouse_result.get('categories', {})
            audits = lighthouse_result.get('audits', {})
            performance_score = categories.get('performance', {}).get('score', 'N/A')
            # Ensure score is scaled 0-100
            if isinstance(performance_score, (float, int)):
                 performance_score = int(performance_score * 100)

            return {
                "Performance_Score": performance_score,
                "First_Contentful_Paint": audits.get('first-contentful-paint', {}).get('displayValue', 'N/A'),
                "Speed_Index": audits.get('speed-index', {}).get('displayValue', 'N/A'),
                "Largest_Contentful_Paint": audits.get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
                "Cumulative_Layout_Shift": audits.get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
                "Time_to_Interactive": audits.get('interactive', {}).get('displayValue', 'N/A'),
                "Total_Blocking_Time": audits.get('total-blocking-time', {}).get('displayValue', 'N/A')
            }

        return {
            "desktop": extract_metrics(desktop_data),
            "mobile": extract_metrics(mobile_data),
            "recommendation": "Analyze both desktop and mobile scores. Focus on improving Core Web Vitals (LCP, CLS, TBT/FID) and addressing specific audit recommendations from Google PageSpeed Insights."
            }

    except requests.exceptions.Timeout:
         return {"error": "Failed to fetch page speed: API request timed out."}
    except requests.exceptions.RequestException as e:
        # Check for specific API errors if possible from response content
        error_details = ""
        try:
            if e.response:
                 error_data = e.response.json()
                 error_details = error_data.get("error", {}).get("message", str(e))
            else:
                 error_details = str(e)
        except Exception: # Fallback if response isn't JSON
             error_details = str(e)
        return {"error": f"Failed to fetch page speed: {error_details}"}
    except Exception as e:
         return {"error": f"An unexpected error occurred during page speed analysis: {str(e)}"}


# 6. Meta Tags Check (V2 - Simplified, relies on Keyword Summary for suggestions)
def check_meta_tags_v2(content):
    """Checks for essential meta tags: title, description, keywords (optional)."""
    soup = BeautifulSoup(content, 'html.parser')
    results = {}
    issues = []

    # Title Tag
    title_tag = soup.find('title')
    if title_tag and title_tag.string:
        results['title'] = title_tag.string.strip()
        if len(results['title']) > 60:
            issues.append(f"Title tag is too long ({len(results['title'])} chars). Aim for 50-60 characters.")
        if len(results['title']) < 20:
             issues.append(f"Title tag seems short ({len(results['title'])} chars). Ensure it's descriptive.")
    else:
        results['title'] = None
        issues.append("Missing Title tag. Essential for SEO.")

    # Meta Description
    meta_desc = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)}) # Case-insensitive name
    if meta_desc and meta_desc.get('content'):
        results['meta_description'] = meta_desc['content'].strip()
        if len(results['meta_description']) > 160:
            issues.append(f"Meta Description is too long ({len(results['meta_description'])} chars). Aim for 150-160 characters.")
        if len(results['meta_description']) < 70:
            issues.append(f"Meta Description seems short ({len(results['meta_description'])} chars). Ensure it's compelling and descriptive.")
    else:
        results['meta_description'] = None
        issues.append("Missing Meta Description. Important for click-through rates from search results.")

    # Meta Keywords (Less important now, but check if present)
    meta_keywords = soup.find('meta', attrs={'name': re.compile(r'^keywords$', re.I)})
    if meta_keywords and meta_keywords.get('content'):
        results['meta_keywords'] = [kw.strip() for kw in meta_keywords['content'].split(',') if kw.strip()]
        # Note: Relevance check against content is complex and better handled by overall keyword analysis
        # issues.append("Meta Keywords tag found. Note: Google largely ignores this tag for ranking, but it might be used by other systems.")
    else:
        results['meta_keywords'] = None
        # issues.append("Meta Keywords tag not found (generally not essential for Google).")


     # Viewport Check
    meta_viewport = soup.find('meta', attrs={'name': re.compile(r'^viewport$', re.I)})
    if not meta_viewport or not meta_viewport.get('content'):
         results['meta_viewport'] = None
         issues.append("Missing Meta Viewport tag. Essential for mobile responsiveness.")
    else:
         results['meta_viewport'] = meta_viewport.get('content')


    return {
        'found_tags': results,
        'issues': issues,
        'recommendation': "Ensure Title and Meta Description tags exist, are within optimal length limits, and accurately describe the page content. Include the primary keyword naturally. Ensure Meta Viewport tag is present for mobile usability. Keyword suggestions are provided in the 'Keyword Summary' section."
    }

# `suggest_keywords` function removed as it's replaced by the logic in `analyze_keyword_summary_v2`


# 7. Broken URL Detection (Modified to accept content)
def detect_broken_urls(url, page_content, max_threads=10):
    """Detects broken URLs using page content to avoid re-fetching."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Function to check a single link
    def check_link_status(link_to_check):
        try:
            # Use HEAD request for efficiency, follow redirects, short timeout
            response = requests.head(link_to_check, headers=headers, allow_redirects=True, timeout=7)
            # Consider 4xx and 5xx errors as broken, allow redirects (3xx should resolve)
            if response.status_code >= 400:
                return (link_to_check, response.status_code) # Broken URL with status code
            return (link_to_check, None) # Valid URL
        except requests.exceptions.Timeout:
             return (link_to_check, "Timeout") # Broken due to timeout
        except requests.exceptions.RequestException as e:
             # Log specific connection errors etc.
             error_type = type(e).__name__
             return (link_to_check, f"Request Error: {error_type}") # Broken URL with error message
        except Exception as e: # Catch any other unexpected errors
             return (link_to_check, f"Unexpected Error: {str(e)}")


    broken_urls_list = []
    valid_urls_list = []
    processed_urls = set() # Keep track of URLs already checked

    try:
        soup = BeautifulSoup(page_content, 'html.parser')
        base_url = url # Used for resolving relative links

        # Find all anchor tags with href attribute
        links = soup.find_all('a', href=True)
        urls_to_check = []

        for link in links:
            href = link['href']
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            # Parse the URL
            parsed_url = urlparse(absolute_url)

            # Filter out non-http(s) links, fragments, and already processed URLs
            if parsed_url.scheme in ['http', 'https'] and absolute_url not in processed_urls:
                 urls_to_check.append(absolute_url)
                 processed_urls.add(absolute_url)


        # Check URLs concurrently
        if urls_to_check:
             with ThreadPoolExecutor(max_workers=max_threads) as executor:
                 results = list(executor.map(check_link_status, urls_to_check))

             for link, error_info in results:
                 if error_info:
                     broken_urls_list.append({"url": link, "status": error_info})
                 else:
                     valid_urls_list.append(link)

        return {
            "total_links_checked": len(processed_urls),
            "broken_links_count": len(broken_urls_list),
            "broken_links": broken_urls_list,
            # "valid_links": valid_urls_list, # Usually too long to be useful in JSON output
            "recommendation": f"Found {len(broken_urls_list)} potentially broken links. Review these links and update or remove them to improve user experience and SEO."
        }

    except Exception as e:
        print(f"Error parsing HTML or processing links: {e}")
        return {
             "error": f"Failed during broken link analysis: {str(e)}",
             "broken_links": [],
             "recommendation": "Could not complete broken link analysis due to an error."
             }



# 8. Keyword Summary and Suggestions (V2 - Central analysis point)
def analyze_keyword_summary_v2(page_content):
    """
    Performs comprehensive keyword analysis using visible text and provides SEO suggestions.
    """
    start_time = time.time()
    soup = BeautifulSoup(page_content, 'html.parser')

    # 1. Extract Visible Text
    original_text = get_visible_text(soup)
    if not original_text or len(original_text.strip()) < 50: # Check for minimum content length
        return {
            "error": "Insufficient meaningful text content found on the page for keyword analysis.",
             "analysis_time_seconds": round(time.time() - start_time, 2)
            }

    # 2. Clean Text for Analysis
    cleaned_text_string, cleaned_words_list = clean_text_for_keywords(original_text)
    if not cleaned_words_list:
         return {
            "error": "Text content found, but contained primarily stopwords or non-alphanumeric characters.",
             "analysis_time_seconds": round(time.time() - start_time, 2)
            }


    # 3. Extract Keywords and Calculate Density/Scores
    keywords_data, total_cleaned_words = extract_keywords_combined_v2(
        original_text,
        cleaned_text_string,
        cleaned_words_list,
        top_n=20 # Extract top 20 candidates
        )

    if not keywords_data:
         return {
            "error": "Could not extract significant keywords from the content.",
            "total_words_analyzed": total_cleaned_words,
             "analysis_time_seconds": round(time.time() - start_time, 2)
            }

    # 4. Generate SEO Suggestions based on keywords
    seo_suggestions = generate_seo_suggestions_v2(keywords_data)

    # 5. (Optional) Generate Content Summary (Simple extractive summary)
    summary = ""
    try:
        sentences = sent_tokenize(original_text) # Use original text for readable sentences
        if len(sentences) > 3: # Only summarize if there are enough sentences
             sentence_scores = {}
             # Score sentences based on presence of top 5 keywords
             top_5_kws = list(keywords_data.keys())[:5]
             for sentence in sentences:
                 score = 0
                 sent_lower = sentence.lower()
                 for kw in top_5_kws:
                     if kw in sent_lower:
                         # Use keyword score for weighting sentence score
                         score += keywords_data[kw]['score']
                 if score > 0:
                      sentence_scores[sentence] = score

             # Get top 3 sentences based on score
             summary_sentences = nlargest(3, sentence_scores.items(), key=lambda item: item[1])
             summary = ' '.join([s[0] for s in summary_sentences])
        else:
            summary = "Content too short to summarize meaningfully."

    except Exception as e:
        summary = f"Could not generate summary: {e}"


    # 6. Format Output
    # Sort keywords by score for the final output
    sorted_keywords_output = sorted(keywords_data.items(), key=lambda item: item[1]['score'], reverse=True)

    return {
        "top_keywords": [{"keyword": kw, **data} for kw, data in sorted_keywords_output],
        "total_words_analyzed": total_cleaned_words,
        "content_summary": summary,
        "seo_suggestions": seo_suggestions,
        "analysis_time_seconds": round(time.time() - start_time, 2)
    }



# 9. Anchor Tag Suggestions (No changes needed)
def analyze_anchor_tags(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    anchor_tags = soup.find_all('a', href=True) # Ensure they have href
    issues = []
    generic_texts = {"click here", "read more", "learn more", "here", "link", "download", "more info", ""}
    empty_anchors = 0
    generic_anchors = 0
    total_anchors = len(anchor_tags)

    for anchor in anchor_tags:
        text = anchor.get_text(strip=True)
        href = anchor.get('href', '')
        if not text:
            empty_anchors += 1
            issues.append(f"Empty anchor text for link: {href[:50]}...")
        elif text.lower() in generic_texts:
            generic_anchors += 1
            issues.append(f"Generic anchor text '{text}' used for link: {href[:50]}...")

    recommendation = f"Found {total_anchors} links. {empty_anchors} have empty anchor text and {generic_anchors} use generic text. Replace generic or empty anchor text with descriptive, keyword-rich phrases relevant to the linked page's content. This helps SEO and usability."

    return {
        "total_anchors": total_anchors,
        "empty_anchor_count": empty_anchors,
        "generic_anchor_count": generic_anchors,
        "issues": issues, # List specific issues found
        "recommendation": recommendation
    }


# 10. URL Structure Optimization (Minor improvements)
def analyze_url_structure(url):
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        query = parsed_url.query
        issues = []
        status = "Optimized"

        # Check length (common recommendation: < 75-100 chars, 60 is quite strict)
        if len(url) > 100:
            status = "Warning"
            issues.append(f"URL length ({len(url)} chars) is quite long. Shorter URLs are often preferred.")

        # Check path structure
        if path and path != '/':
             # Check for underscores (hyphens preferred)
             if '_' in path:
                  status = "Warning"
                  issues.append("URL path contains underscores ('_'). Hyphens ('-') are generally preferred for word separation.")
             # Check for excessive parameters in path (less common, but possible)
             if path.count('/') > 5: # Arbitrary depth limit
                  issues.append("URL path seems deep (many '/'). Consider flatter structure if possible.")
             # Check for file extensions (e.g., .html, .php) - often better without
             if re.search(r'\.\w{2,4}$', path): # Matches .xx, .xxx, .xxxx at the end
                 issues.append("URL path includes a file extension. Consider configuring server to remove them for cleaner URLs.")
             # Check for case sensitivity issues (lowercase preferred)
             if any(c.isupper() for c in path):
                  status = "Warning"
                  issues.append("URL path contains uppercase letters. Using lowercase only is recommended to avoid duplicate content issues.")

        # Check query parameters
        if query:
            params = query.split('&')
            if len(params) > 3:
                status = "Warning"
                issues.append(f"URL has {len(params)} query parameters. Excessive parameters can sometimes hinder crawling or indicate non-canonical URLs.")
            # Check for overly long parameters
            if any(len(p) > 50 for p in params):
                 issues.append("URL contains very long query parameters.")

        recommendation = "Aim for short, descriptive, lowercase URLs using hyphens for word separation. Avoid unnecessary parameters or file extensions. Ensure URLs clearly reflect the page hierarchy/content."
        if status != "Optimized":
             recommendation += " Address the specific issues noted above."

        return {
            "url": url,
            "status": status,
            "path": path,
            "query_params": query,
            "issues": issues,
            "recommendation": recommendation
        }
    except Exception as e:
         return {"error": f"Failed to analyze URL structure: {str(e)}"}


# 11. Robots.txt Validation (No changes needed)
def validate_robots_txt(url):
    parsed_url = urlparse(url)
    # Ensure scheme and netloc are present
    if not parsed_url.scheme or not parsed_url.netloc:
        return {"error": "Invalid URL provided for robots.txt check."}

    robots_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", "/robots.txt")
    status = "Unknown"
    content = None
    issues = []
    recommendation = ""

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'} # Simulate Googlebot
        response = requests.get(robots_url, headers=headers, timeout=10)

        if response.status_code == 200:
            status = "Exists"
            content = response.text
            # Basic validation checks
            if not content.strip():
                 issues.append("robots.txt exists but is empty.")
            if "User-agent: *" not in content and "User-agent: Googlebot" not in content:
                 issues.append("No specific rules found for major crawlers like '*' or 'Googlebot'.")
            # Check for potential blocking of important resources (basic check)
            if re.search(r"Disallow:\s*/\s*$", content, re.MULTILINE):
                 issues.append("Found 'Disallow: /', which blocks the entire site. Verify this is intended.")
            recommendation = "robots.txt found. Review its rules to ensure necessary content is crawlable and sensitive areas are disallowed. Check Google Search Console for crawl errors related to robots.txt."

        elif response.status_code == 404:
            status = "Missing"
            recommendation = "No robots.txt file found (404 error). Create one to guide search engine crawlers. If you want everything crawled, an empty file or one allowing all is fine."
        else:
            status = f"Error ({response.status_code})"
            recommendation = f"Could not fetch robots.txt. Server returned status code {response.status_code}. Ensure the file is accessible."

    except requests.exceptions.Timeout:
         status = "Error (Timeout)"
         recommendation = "Request timed out trying to fetch robots.txt."
    except requests.exceptions.RequestException as e:
        status = f"Error (Request Failed)"
        recommendation = f"Failed to fetch robots.txt: {str(e)}. Check DNS or network connectivity."
    except Exception as e:
        status = f"Error (Unexpected)"
        recommendation = f"An unexpected error occurred checking robots.txt: {str(e)}"


    return {
        "robots_url": robots_url,
        "status": status,
        "content": content, # Include content only if found
        "issues": issues,
        "recommendation": recommendation
        }


# 12. XML Sitemap Validation (Basic check)
def validate_sitemap(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return {"error": "Invalid URL provided for sitemap check."}

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    # Common sitemap locations
    sitemap_locations = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"), # Common for index files
        urljoin(base_url, "/sitemap.php"), # Less common, but possible
        urljoin(base_url, "/sitemap.txt"), # Plain text sitemap
    ]

    found_sitemap_url = None
    status = "Missing"
    error_message = None
    content_preview = None

    # Check robots.txt first for sitemap directive
    robots_info = validate_robots_txt(url) # Reuse robots check
    if robots_info and 'content' in robots_info and robots_info['content']:
         sitemap_directives = re.findall(r'^\s*Sitemap:\s*(.*)', robots_info['content'], re.IGNORECASE | re.MULTILINE)
         if sitemap_directives:
              # Add URLs from robots.txt to the beginning of the check list
              sitemap_locations = [url.strip() for url in sitemap_directives] + sitemap_locations
              sitemap_locations = list(dict.fromkeys(sitemap_locations)) # Deduplicate


    headers = {'User-Agent': 'Mozilla/5.0'}
    for sitemap_url in sitemap_locations:
        try:
            response = requests.get(sitemap_url, headers=headers, timeout=10)
            if response.status_code == 200:
                 # Basic check if it looks like XML (or text for .txt)
                 content_type = response.headers.get('Content-Type', '').lower()
                 is_xml = 'xml' in content_type
                 is_text = 'text/plain' in content_type
                 is_likely_sitemap = is_xml or (is_text and sitemap_url.endswith('.txt'))

                 if is_likely_sitemap and len(response.content) > 10: # Check if not empty
                    status = "Exists"
                    found_sitemap_url = sitemap_url
                    content_preview = response.text[:200] + "..." # Get a preview
                    break # Stop after finding the first valid one
            # Don't set error message yet, just continue checking other locations
            # elif response.status_code != 404:
            #      # Log other status codes if needed
            #      pass

        except requests.exceptions.Timeout:
             error_message = f"Timeout checking {sitemap_url}" # Record last error
        except requests.exceptions.RequestException as e:
            error_message = f"Request Error checking {sitemap_url}: {type(e).__name__}" # Record last error
        except Exception as e:
            error_message = f"Unexpected error checking {sitemap_url}: {str(e)}" # Record last error

    recommendation = "An XML sitemap helps search engines discover all your important pages. Create one and submit it via Google Search Console. Ensure it's listed in your robots.txt file."
    if status == "Exists":
         recommendation = f"Sitemap found at {found_sitemap_url}. Ensure it's up-to-date, error-free (use validator tools), and submitted to search engines. Check Google Search Console for coverage status."
    elif error_message:
         recommendation += f" An error occurred during the check: {error_message}"


    return {
        "status": status,
        "checked_locations": sitemap_locations, # Show where it looked
        "found_sitemap_url": found_sitemap_url,
        "content_preview": content_preview,
        "error_message": error_message,
        "recommendation": recommendation
    }


# 13. Blog Optimization (Requires `googlesearch` or alternative, kept structure but needs attention if used)
# Note: This function uses the `googlesearch` library which can be unreliable and might get blocked.
# Consider replacing with a different approach or official API if available.
def analyze_blog_optimization(page_content, url):
    """Analyzes blog title, H1s, and suggests improvements based on competitors."""
    # !!! This function depends on the `googlesearch` library. Ensure it's installed
    # and be aware of potential rate limiting or blocking by Google. !!!
    try:
        from googlesearch import search
    except ImportError:
        return {"error": "The 'google' package is required for blog optimization analysis but is not installed. Skipping."}


    soup = BeautifulSoup(page_content, 'html.parser')
    title = soup.find('title').get_text(strip=True) if soup.find('title') else "No title found"
    h1_tags = [h1.get_text(strip=True) for h1 in soup.find_all('h1') if h1.get_text(strip=True)]
    main_h1 = h1_tags[0] if h1_tags else None

    if not main_h1:
        return {
            "current_title": title,
            "h1_tags": h1_tags,
            "warning": "No suitable H1 tag found for competitive title analysis.",
            "summary": "Ensure the blog post has a clear H1 heading representing the main topic."
        }

    competitor_titles = []
    google_suggested_titles = []
    error_messages = []

    # --- Competitor Analysis based on H1 ---
    try:
        # Search Google for the main H1 topic
        query = f'"{main_h1}" blog post' # Search for the specific H1 phrase
        print(f"Searching Google for: {query}")
        # Use stop=5 instead of num_results, add pause to be polite
        search_results = list(search(query, num=5, stop=5, pause=2.0, lang="en"))

        # Fetch titles from the search results
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_url = {executor.submit(requests.get, result_url, timeout=7, headers={'User-Agent': 'Mozilla/5.0'}): result_url for result_url in search_results if url not in result_url} # Exclude self

            for future in as_completed(future_to_url):
                result_url = future_to_url[future]
                try:
                    response = future.result()
                    response.raise_for_status()
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        result_soup = BeautifulSoup(response.text, 'html.parser')
                        result_title = result_soup.find('title').get_text(strip=True) if result_soup.find('title') else None
                        if result_title:
                            google_suggested_titles.append(result_title)
                except requests.exceptions.RequestException as e:
                     error_messages.append(f"Error fetching competitor title from {result_url}: {type(e).__name__}")
                except Exception as e:
                     error_messages.append(f"Error processing competitor {result_url}: {str(e)}")

    except ImportError:
         error_messages.append("Skipping Google search: 'google' package not installed.")
    except Exception as e:
         # Catch potential errors from googlesearch library (e.g., rate limiting)
         error_messages.append(f"Error during Google search for competitors: {str(e)}")


    # --- Generate a Suggested Title ---
    # Basic suggestion: Combine original title elements with common keywords from competitors
    suggested_title = title # Default to original
    if google_suggested_titles:
         # Extract common words (excluding stopwords) from competitor titles
         all_comp_words = []
         comp_text = ' '.join(google_suggested_titles)
         cleaned_comp_text, comp_words_list = clean_text_for_keywords(comp_text)
         if comp_words_list:
              word_counts = Counter(comp_words_list)
              # Get top 2-3 most frequent non-stop words from competitor titles
              top_comp_keywords = [kw[0] for kw in word_counts.most_common(3) if kw[0] not in ENGLISH_STOP_WORDS]
              if top_comp_keywords:
                  # Simple combination - might need more sophisticated logic
                   suggested_title = f"{main_h1}: Guide with {', '.join(top_comp_keywords).title()}"


    return {
        "current_title": title,
        "h1_tags": h1_tags,
        "analysis_based_on_h1": main_h1,
        "competitor_titles_found": len(google_suggested_titles),
        # "google_suggested_titles": google_suggested_titles, # Can be very long
        "suggested_title_based_on_competitors": suggested_title,
        "errors": error_messages,
        "summary": "Analyze competitor titles for similar topics (using H1 as a base). Consider incorporating common relevant terms found in top-ranking competitor titles into your own title and headings, while keeping it unique and compelling."
    }

# --- Index View ---
def index(request):
    return render(request, 'index.html') # Make sure 'index.html' exists in your templates folder