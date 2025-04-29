
from django.http import JsonResponse
from django.shortcuts import render
# Removed unused import: from pypsrp import FEATURES
# Removed unused import: from .models import * # Assuming models aren't used in this specific view
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse, urljoin
import os
from openai import OpenAI # Keep if needed elsewhere, but not used in provided functions
# client = OpenAI(api_key='YOUR_SECURE_API_KEY') # Use environment variables for keys!

from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams # For keyword phrase extraction
from collections import Counter
# Removed unused import: from heapq import nlargest # Replaced with Counter.most_common
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Removed unreliable scraping: from googlesearch import search
from transformers import T5ForConditionalGeneration, T5Tokenizer # For AI detection

# --- NLTK Data Downloads (Ensure these run successfully) ---
# It's better to run these once during setup/deployment
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
# -----------------------------------------------------------

# --- Constants ---
# Use environment variables for API keys in production!
PAGESPEED_API_KEY = os.environ.get('PAGESPEED_API_KEY', 'AIzaSyCzMj9hqnN8lSmMIc2vMQZ2mC9N-AcNvcQ') # Example fallback
PAGESPEED_TIMEOUT = 90
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
REQUEST_TIMEOUT = 10 # Timeout for external requests in seconds

# --- Main View ---
def index(request):
    return render(request, 'index.html')

def analyze_page(request):
    start_time = time.time() # Record the start time

    if request.method == 'GET':
        url = request.GET.get('url')

        if not url or not url.startswith(('http://', 'https://')):
            return JsonResponse({"error": "A valid URL starting with http:// or https:// is required."})

        # Validate URL format roughly
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL structure")
        except ValueError as e:
            return JsonResponse({"error": f"Invalid URL format: {str(e)}"})


        # Fetch the page content
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            page_content = response.text
            content_type = response.headers.get('Content-Type', '').lower()
            if 'html' not in content_type:
                 return JsonResponse({"error": f"URL does not point to an HTML page (Content-Type: {content_type})."})

        except requests.exceptions.Timeout:
             return JsonResponse({"error": f"Failed to fetch the page: Request timed out after {REQUEST_TIMEOUT} seconds."})
        except requests.exceptions.RequestException as e:
             return JsonResponse({"error": f"Failed to fetch the page: {str(e)}"})
        except Exception as e:
             return JsonResponse({"error": f"An unexpected error occurred while fetching the page: {str(e)}"})

        analysis_results = {}

        # Use ThreadPoolExecutor to run analyses in parallel
        with ThreadPoolExecutor(max_workers=10) as executor: # Adjust max_workers based on resources
            # Submit tasks - pass content or URL as needed
            future_to_analysis = {
                executor.submit(analyze_on_page_optimization, page_content): 'on_page_optimization',
                executor.submit(analyze_h1_tag, page_content): 'h1_tag',
                executor.submit(validate_schema, page_content): 'schema_validation',
                executor.submit(detect_ai_content, page_content): 'ai_content_detection', # Potentially slow
                executor.submit(analyze_page_speed, url): 'page_speed',
                executor.submit(check_meta_tags, page_content): 'meta_tags',
                executor.submit(analyze_keyword_summary, page_content): 'keyword_summary', # Enhanced
                executor.submit(analyze_anchor_tags, page_content): 'anchor_tags',
                executor.submit(analyze_url_structure, url): 'url_structure',
                executor.submit(validate_robots_txt, url): 'robots_txt',
                executor.submit(validate_sitemap, url): 'xml_sitemap',
                executor.submit(analyze_blog_optimization, page_content, url): 'blog_optimization', # Reworked
                executor.submit(detect_broken_urls, url): 'detect_broken_urls', # Returns only broken
                executor.submit(analyze_html_structure, page_content): 'html_structure', # New
                executor.submit(analyze_da_pa_spam, url): 'da_pa_spam_score' # New (Placeholder)
            }

            for future in as_completed(future_to_analysis):
                analysis_name = future_to_analysis[future]
                try:
                    result = future.result()
                    analysis_results[analysis_name] = result
                except Exception as e:
                    # Log the error for debugging
                    print(f"Error in analysis '{analysis_name}': {str(e)}")
                    analysis_results[analysis_name] = {"error": f"Analysis failed: {str(e)}"}

        end_time = time.time() # Record the end time
        execution_time = end_time - start_time # Calculate the difference

        print(f"Analysis for {url} completed in {execution_time:.2f} seconds")

        # Combine keyword results if needed (optional)
        # For instance, pass keywords from analyze_keyword_summary to other functions
        # if they require them and weren't passed initially.
        # Example: analysis_results['on_page_optimization']['keyword_density'] = analysis_results['keyword_summary'].get('keyword_density', {})

        return JsonResponse(analysis_results)

    return JsonResponse({"error": "Invalid request method. Use GET."})


# --- Analysis Functions ---

# 1. On-Page Optimization (Headings)
def analyze_on_page_optimization(content):
    """
    Analyzes heading structure (H1-H6). Keyword density is now handled in analyze_keyword_summary.
    """
    soup = BeautifulSoup(content, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    heading_hierarchy = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []}
    issues = []
    suggestions = []

    for heading in headings:
        tag = heading.name
        text = heading.text.strip()
        if text: # Only consider headings with text
            heading_hierarchy[tag].append(text)

    # Check hierarchy logic
    last_level = 0
    for i in range(1, 7):
        tag = f'h{i}'
        if heading_hierarchy[tag]:
            current_level = i
            if current_level > last_level + 1:
                 issues.append(f"Heading level skipped: Found <{tag}> after <h{last_level}>. Should follow a logical order (e.g., H1 -> H2 -> H3).")
            last_level = current_level
        elif last_level >= i : # If we found e.g. H3 but no H2 later
            pass # This case is less critical than skipping

    if not heading_hierarchy['h1']:
        issues.append("Missing H1 tag. Every page should ideally have one primary H1 heading.")
    elif len(heading_hierarchy['h1']) > 1:
        issues.append("Multiple H1 tags found. Best practice is to use only one H1 per page.")

    if issues:
        suggestions.append("Review heading structure. Ensure headings form a logical outline (H1 > H2 > H3...) without skipping levels. Use headings to structure content clearly for users and search engines.")
    else:
        suggestions.append("Heading structure appears logical. Ensure headings accurately reflect the content of their sections.")

    suggestions.append("Consider incorporating primary keywords naturally within your H1 and H2 tags where relevant.")

    return {
        'headings': heading_hierarchy,
        'issues': issues,
        'suggestions': suggestions
        # Keyword density removed - handled by analyze_keyword_summary
    }

# (Removed standalone get_keyword_density - integrated into analyze_keyword_summary)

# 2. H1 Tag Check
def analyze_h1_tag(content):
    """
    Analyzes the H1 tag specifically against the title.
    """
    soup = BeautifulSoup(content, 'html.parser')
    h1_tags = soup.find_all('h1')
    title_tag = soup.find('title')

    h1_texts = [h1.text.strip() for h1 in h1_tags if h1.text.strip()]
    title_text = title_tag.text.strip() if title_tag else None

    issues = []
    suggestions = []

    if not h1_texts:
        issues.append("No H1 tag found.")
        suggestions.append("Add a primary H1 tag that clearly describes the main topic of the page. It's crucial for SEO and accessibility.")
    elif len(h1_texts) > 1:
        issues.append(f"Multiple H1 tags found ({len(h1_texts)}).")
        suggestions.append("Use only one H1 tag per page to define the main heading. Convert other H1s to H2s or lower if appropriate.")

    if not title_text:
        issues.append("No <title> tag found.")
        suggestions.append("Add a concise and descriptive <title> tag. It appears in browser tabs and search results.")
    elif h1_texts and title_text:
        # Check similarity (simple comparison, could be improved with NLP)
        main_h1 = h1_texts[0]
        if main_h1.lower() != title_text.lower():
             suggestions.append(f"The primary H1 ('{main_h1}') and the Title ('{title_text}') differ. While not strictly required, ensuring they are closely related and both contain target keywords can improve relevance signals.")
        else:
             suggestions.append("The primary H1 and Title tags are identical or very similar. This is generally good practice.")

    return {
        'h1_found': len(h1_texts) > 0,
        'h1_count': len(h1_texts),
        'h1_texts': h1_texts,
        'title_text': title_text,
        'issues': issues,
        'suggestions': suggestions
    }

# 3. Schema.org Validation
def validate_schema(page_content):
    """
    Finds JSON-LD schema.org data and identifies its types.
    Does not perform full validation against schema.org standards.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    scripts = soup.find_all('script', type='application/ld+json')
    schemas = []
    parsing_errors = []

    for script in scripts:
        try:
            # Handle potential comments within script tags if necessary
            script_content = script.string
            if script_content:
                # Basic cleaning of common non-JSON elements if needed (use cautiously)
                # script_content = re.sub(r'^\s*//.*$', '', script_content, flags=re.MULTILINE) # Remove // comments
                data = json.loads(script_content)
                if isinstance(data, list):
                    schemas.extend(data)
                else:
                    schemas.append(data)
        except json.JSONDecodeError as e:
            parsing_errors.append(f"Could not parse JSON-LD: {e}. Content: {script.string[:100]}...")
        except Exception as e:
            parsing_errors.append(f"Unexpected error parsing script: {e}")

    results = []
    schema_types_found = set()
    suggestions = []

    for schema in schemas:
        if isinstance(schema, dict):
            schema_type = schema.get('@type')
            if schema_type:
                results.append({"type": schema_type, "content": schema})
                schema_types_found.add(schema_type)
            else:
                parsing_errors.append("Found JSON-LD object missing '@type'.")
        # else: # Handle cases where top-level isn't a dict, though less common for schema
            # parsing_errors.append(f"Found JSON-LD element that is not an object: {type(schema)}")


    if not results:
        suggestions.append("No Schema.org structured data (JSON-LD) found. Adding structured data helps search engines understand your content better and can enable rich results (e.g., ratings, FAQs).")
    else:
        suggestions.append(f"Found Schema.org types: {', '.join(schema_types_found)}. Verify the implementation using Google's Rich Results Test tool.")
        suggestions.append("Consider adding more specific schema types relevant to your content (e.g., Article, Product, LocalBusiness, FAQPage) if applicable.")

    return {
        "schemas_found": results,
        "parsing_errors": parsing_errors,
        "suggestions": suggestions
        }



# 4. AI Content Detection (using Transformers T5)
def preprocess_content(content):
    """
    Improved preprocessing to extract cleaner text content from HTML.
    """
    soup = BeautifulSoup(content, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text, separating paragraphs and block elements better
    text = soup.get_text(separator='\n', strip=True)

    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n', '\n', text) # Collapse multiple newlines
    text = re.sub(r'[ \t]+', ' ', text) # Collapse spaces/tabs within lines
    text = text.strip()

    # Basic punctuation cleaning (optional, depending on model needs)
    # text = re.sub(r'([.,!?])\1+', r'\1', text) # Remove repeated punctuation

    return text

# Cache for the T5 model and tokenizer to avoid reloading on every request
t5_model = None
t5_tokenizer = None

def load_t5_model():
    """Loads the T5 model and tokenizer if not already loaded."""
    global t5_model, t5_tokenizer
    if t5_model is None or t5_tokenizer is None:
        try:
            print("Loading T5 model and tokenizer (this may take a moment)...")
            # Consider using 'google/flan-t5-small' or other variants if 't5-small' has issues
            model_name = "t5-small"
            t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            print("T5 model loaded.")
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            # Set to False to prevent retries within the same run if loading fails
            t5_model = False
            t5_tokenizer = False
    return t5_model, t5_tokenizer


def detect_ai_content(content):
    """
    Detects potential AI-like patterns and suggests rewrites using T5.
    NOTE: T5 inference can be slow and resource-intensive.
    """
    performance_note = "AI detection using T5 model can be slow."
    clean_content = preprocess_content(content)

    if len(clean_content.strip()) < 50: # Need sufficient text
        return {
            "performance_note": performance_note,
            "ai_detected_heuristic": False,
            "readability_score": None,
            "paraphrase_suggestion": "Content too short for reliable AI analysis or paraphrasing.",
            "error": None
        }

    ai_detected_heuristic = False
    readability_score = None
    human_suggestion = "Could not generate paraphrase suggestion."
    error_message = None

    # --- Heuristics (Basic Checks) ---
    # Generic phrases often found in basic AI output (expand this list)
    generic_phrases = [
        "in conclusion", "it is important to note", "as an AI language model",
        "unlock the power", "delve into the world", "in the digital age"
    ]
    for phrase in generic_phrases:
        if phrase in clean_content.lower():
            ai_detected_heuristic = True
            break

    # Check readability score (very high/low scores *might* indicate non-human patterns)
    try:
        readability_score = flesch_reading_ease(clean_content)
        # Adjust thresholds based on expected content type
        if readability_score < 30 or readability_score > 85:
             # This is a weak indicator, use with caution
             # ai_detected_heuristic = True
             pass
    except Exception as e:
        error_message = f"Error calculating readability score: {e}"

    # --- Paraphrasing with T5 (Slow Part) ---
    try:
        model, tokenizer = load_t5_model()

        if model and tokenizer: # Check if loading was successful
            # Paraphrase a segment rather than the whole text for speed/relevance
            # Taking the first ~200 words as an example segment
            segment_to_paraphrase = " ".join(clean_content.split()[:200])

            input_text = f"paraphrase: {segment_to_paraphrase}"
            # Ensure tokenizer handles potential long inputs correctly
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

            # Generate paraphrased text
            outputs = model.generate(
                input_ids,
                max_length=512, # Max length of generated output
                num_return_sequences=1,
                num_beams=4,      # Beam search for potentially better quality
                early_stopping=True # Stop when end token is generated
                # temperature=0.7 # Adjust creativity vs coherence if needed
            )
            human_suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif model is False: # Loading failed previously
             error_message = "T5 model failed to load, cannot generate paraphrase."
             human_suggestion = "Paraphrasing unavailable due to model loading error."
        else:
             # Should not happen if load_t5_model logic is correct
             error_message = "T5 model/tokenizer not available."
             human_suggestion = "Paraphrasing unavailable."

    except Exception as e:
        error_message = f"Error during T5 paraphrase generation: {str(e)}"
        human_suggestion = f"Could not generate paraphrase suggestion due to error: {str(e)}"


    # --- Suggestions ---
    suggestions = []
    if ai_detected_heuristic:
        suggestions.append("Content shows some patterns (e.g., generic phrases, unusual readability score) that *might* indicate AI generation or unnatural writing. Review for authenticity and clarity.")
    else:
         suggestions.append("Heuristic checks did not find strong indicators of AI generation. However, manual review is always recommended.")

    if readability_score is not None:
         suggestions.append(f"Flesch Reading Ease score: {readability_score:.2f}. Aim for a score appropriate for your target audience (e.g., 60-70 for general public).")

    suggestions.append("Consider the generated paraphrase suggestion for alternative phrasing, but always review and edit suggestions to ensure accuracy, tone, and originality.")


    return {
        "performance_note": performance_note,
        "ai_detected_heuristic": ai_detected_heuristic,
        "readability_score": f"{readability_score:.2f}" if readability_score is not None else "N/A",
        "paraphrase_suggestion": human_suggestion,
        "suggestions": suggestions,
        "error": error_message
    }


# 5. Page Speed Analysis
def analyze_page_speed(url):
    """
    Analyzes page speed using Google PageSpeed Insights API.
    """
    # Ensure API key is available
    if not PAGESPEED_API_KEY or PAGESPEED_API_KEY == 'YOUR_SECURE_API_KEY':
        return {"error": "PageSpeed API key is missing or not configured."}

    api_url = f'https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={PAGESPEED_API_KEY}&category=PERFORMANCE' # Focus on performance

    try:
        print("Requesting Pagespeed analysis for {url} with timeout {PAGESPEED_TIMEOUT}s...")
        response = requests.get(api_url, timeout=PAGESPEED_TIMEOUT) # Allow longer timeout for API
        response.raise_for_status()
        data = response.json()

        lighthouse_result = data.get('lighthouseResult', {})
        categories = lighthouse_result.get('categories', {})
        audits = lighthouse_result.get('audits', {})
        performance_score = categories.get('performance', {}).get('score')

        # Extract core web vitals and other key metrics
        metrics = {
            "Performance_Score": f"{int(performance_score * 100)}" if performance_score is not None else "N/A",
            "First_Contentful_Paint": audits.get('first-contentful-paint', {}).get('displayValue', 'N/A'),
            "Speed_Index": audits.get('speed-index', {}).get('displayValue', 'N/A'),
            "Largest_Contentful_Paint": audits.get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
            "Cumulative_Layout_Shift": audits.get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
            "Time_to_Interactive": audits.get('interactive', {}).get('displayValue', 'N/A'),
            "Total_Blocking_Time": audits.get('total-blocking-time', {}).get('displayValue', 'N/A')
        }

        # Add suggestions based on score
        suggestions = []
        if performance_score is not None:
            score_num = int(performance_score * 100)
            if score_num < 50:
                suggestions.append(f"Performance score ({score_num}) is poor. Focus on critical optimizations like image compression, reducing JavaScript execution time, and server response time.")
            elif score_num < 90:
                suggestions.append(f"Performance score ({score_num}) is average. Address specific audit suggestions from PageSpeed Insights to improve metrics like LCP and TBT.")
            else:
                suggestions.append(f"Performance score ({score_num}) is good! Continue monitoring Core Web Vitals.")
        else:
            suggestions.append("Could not retrieve performance score. Check PageSpeed Insights directly for detailed audits.")

        suggestions.append("Prioritize improving Largest Contentful Paint (LCP), Cumulative Layout Shift (CLS), and potentially First Input Delay (FID) / Interaction to Next Paint (INP) - closely related to TBT.")
        suggestions.append("Use tools like Google PageSpeed Insights website, Lighthouse (in Chrome DevTools), and WebPageTest.org for detailed diagnostics.")

        return {
            "metrics": metrics,
            "suggestions": suggestions
            }

    except requests.exceptions.Timeout:
         return {"error": f"PageSpeed API request timed out."}
    except requests.exceptions.RequestException as e:
        # Try to parse error response from Google API
        error_details = ""
        try:
             error_data = e.response.json()
             error_details = error_data.get('error', {}).get('message', '')
        except:
             pass # Ignore if response is not JSON
        return {"error": f"Failed to fetch page speed: {str(e)}. {error_details}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during page speed analysis: {str(e)}"}


# 6. Meta Tags Check
def check_meta_tags(content):
    """
    Checks for meta description and keywords (though keywords are less important now).
    Relies on analyze_keyword_summary for keyword suggestions.
    """
    soup = BeautifulSoup(content, 'html.parser')

    meta_desc_tag = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
    meta_keywords_tag = soup.find('meta', attrs={'name': re.compile(r'^keywords$', re.I)})
    # Also check Open Graph tags (important for social sharing)
    og_title = soup.find('meta', property='og:title')
    og_description = soup.find('meta', property='og:description')
    og_image = soup.find('meta', property='og:image')

    meta_description = meta_desc_tag['content'].strip() if meta_desc_tag and meta_desc_tag.get('content') else None
    meta_keywords = meta_keywords_tag['content'].strip() if meta_keywords_tag and meta_keywords_tag.get('content') else None

    issues = []
    suggestions = []

    # Meta Description
    if not meta_description:
        issues.append("Meta description is missing.")
        suggestions.append("Add a unique and compelling meta description (around 150-160 characters) that accurately summarizes the page content and encourages clicks from search results.")
    elif len(meta_description) < 70:
        issues.append("Meta description is very short.")
        suggestions.append("Expand the meta description to provide more context and include relevant keywords (aim for 150-160 characters).")
    elif len(meta_description) > 165: # Approximate limit
        issues.append("Meta description might be too long and could be truncated in search results.")
        suggestions.append("Shorten the meta description to ensure the most important information is visible (aim for 150-160 characters).")
    else:
        suggestions.append("Meta description length seems appropriate. Ensure it is unique, descriptive, and includes target keywords.")

    # Meta Keywords
    if meta_keywords:
        suggestions.append("Meta keywords tag found. Note: Google and most major search engines largely ignore this tag for ranking purposes. Focus efforts on content quality and other meta tags instead.")
    else:
        suggestions.append("Meta keywords tag is missing. This is generally fine as it's not a significant ranking factor for major search engines.")

    # Open Graph Tags
    if not og_title or not og_title.get('content'):
         issues.append("Open Graph Title (og:title) is missing.")
         suggestions.append("Add an 'og:title' meta tag for optimal display when shared on social media platforms.")
    if not og_description or not og_description.get('content'):
         issues.append("Open Graph Description (og:description) is missing.")
         suggestions.append("Add an 'og:description' meta tag for social media sharing.")
    if not og_image or not og_image.get('content'):
         issues.append("Open Graph Image (og:image) is missing.")
         suggestions.append("Add an 'og:image' meta tag to specify the image used when the page is shared on social media.")


    return {
        'meta_description': meta_description,
        'meta_keywords': meta_keywords,
        'open_graph_tags': {
            'og:title': og_title['content'].strip() if og_title and og_title.get('content') else None,
            'og:description': og_description['content'].strip() if og_description and og_description.get('content') else None,
            'og:image': og_image['content'].strip() if og_image and og_image.get('content') else None,
        },
        'issues': issues,
        'suggestions': suggestions
        # Keyword suggestions are now part of analyze_keyword_summary
    }

# (Removed suggest_keywords function - logic merged into analyze_keyword_summary)


# 7. Broken URL Detection
def detect_broken_urls(url, max_threads=10):
    """
    Detects broken internal and external links on the page. Returns only broken URLs.
    """
    broken_urls_info = []
    processed_urls = set() # Avoid checking the same URL multiple times

    def check_link_status(link_to_check):
        """Helper function to check URL status."""
        if link_to_check in processed_urls:
            return None # Already checked

        processed_urls.add(link_to_check)

        try:
            # Use HEAD request for efficiency, fallback to GET if HEAD fails/is disallowed
            response = requests.head(link_to_check, headers=HEADERS, allow_redirects=True, timeout=REQUEST_TIMEOUT)
            # Some servers block HEAD, try GET
            if not response.ok: # Status code >= 400 or other issue with HEAD
                 time.sleep(0.2) # Small delay before GET
                 response = requests.get(link_to_check, headers=HEADERS, allow_redirects=True, timeout=REQUEST_TIMEOUT)

            if not response.ok: # Check status code after HEAD or GET
                return (link_to_check, response.status_code) # Broken URL with status code
            return None # Valid URL

        except requests.exceptions.Timeout:
            return (link_to_check, "Timeout")
        except requests.exceptions.RequestException as e:
            # Simplify common errors
            error_str = str(e)
            if "SSL" in error_str: return (link_to_check, "SSL Error")
            if "Connection refused" in error_str: return (link_to_check, "Connection Refused")
            if "Name or service not known" in error_str: return (link_to_check, "DNS Error")
            return (link_to_check, f"Request Error: {type(e).__name__}") # Generic error
        except Exception as e: # Catch other potential errors
             return (link_to_check, f"Unexpected Error: {type(e).__name__}")

    try:
        # Fetch the initial page
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all valid http/https links
        links_to_check = set()
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            # Basic validation and normalization
            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                absolute_link = urljoin(url, href) # Resolve relative URLs
                parsed_link = urlparse(absolute_link)
                if parsed_link.scheme in ['http', 'https']:
                    # Remove fragments (#) for checking
                    links_to_check.add(parsed_link._replace(fragment="").geturl())

        # Check links concurrently
        if links_to_check:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_to_link = {executor.submit(check_link_status, link): link for link in links_to_check}
                for future in as_completed(future_to_link):
                    result = future.result()
                    if result: # If function returned a broken link tuple
                        broken_urls_info.append({"url": result[0], "status": result[1]})
        else:
             # No links found to check
             pass


        suggestions = []
        if broken_urls_info:
            suggestions.append(f"Found {len(broken_urls_info)} potentially broken links. Broken links negatively impact user experience and can waste crawl budget.")
            suggestions.append("Review the listed URLs and update or remove them. Check both internal and external links.")
        else:
            suggestions.append("No broken links detected in this scan. Regularly check for broken links as part of website maintenance.")

        return {
            "broken_urls": broken_urls_info, # Only return broken ones
            "suggestions": suggestions
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page ({url}) for link checking: {e}")
        return {"broken_urls": [], "error": f"Could not fetch the page to check links: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error during broken link detection for {url}: {e}")
        return {"broken_urls": [], "error": f"An unexpected error occurred during link checking: {str(e)}"}


# 8. Keyword Analysis and Summary (Enhanced)
def clean_text_for_keywords(text):
    """ More robust text cleaning for keyword analysis. """
    text = text.lower()
    # Remove HTML tags (redundant if using soup.get_text, but safe)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove punctuation and numbers, keep spaces and basic word chars
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    return [" ".join(gram) for gram in ngrams(tokens, n)]

def analyze_keyword_summary(page_content):
    """
    Enhanced keyword analysis using NLTK, n-grams, frequency, and TF-IDF.
    Calculates keyword density and generates a summary.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    # Extract text from meaningful tags, exclude nav/footer if possible
    main_content = soup.find('main') or soup.find('article') or soup.find('body') # Prioritize main content areas
    text_content = main_content.get_text(separator=" ", strip=True) if main_content else soup.get_text(separator=" ", strip=True)

    cleaned_text = clean_text_for_keywords(text_content)
    tokens = word_tokenize(cleaned_text)
    stop_words_set = set(stopwords.words('english'))
    # Add custom irrelevant words if needed
    custom_stops = {'use', 'like', 'get', 'also', 'make', 'one', 'may', 'need'}
    stop_words_set.update(custom_stops)

    filtered_tokens = [word for word in tokens if word not in stop_words_set and len(word) > 2]

    if not filtered_tokens or len(filtered_tokens) < 10: # Need enough content
        return {
            "top_keywords": [],
            "keyword_density": {},
            "summary": "Insufficient text content for meaningful keyword analysis.",
            "gist": "Not enough text content.",
            "seo_suggestions": {"suggested_keywords": [], "guidelines": ["Add more relevant content to the page."]} ,
            "error": "Insufficient text content."
            }

    # --- Calculate Term Frequencies (TF) for words and phrases ---
    word_freq = Counter(filtered_tokens)
    bigram_freq = Counter(get_ngrams(filtered_tokens, 2))
    trigram_freq = Counter(get_ngrams(filtered_tokens, 3))

    # Combine frequencies (simple approach: sum counts, could be weighted)
    combined_freq = Counter()
    combined_freq.update({word: count for word, count in word_freq.items()})
    combined_freq.update({gram: count for gram, count in bigram_freq.items() if count > 1}) # Require bigrams to appear > once
    combined_freq.update({gram: count for gram, count in trigram_freq.items() if count > 1}) # Require trigrams to appear > once

    # Get top N keywords/phrases by frequency
    top_n = 15
    top_keywords_freq = dict(combined_freq.most_common(top_n))

    # --- Calculate Keyword Density ---
    total_words = len(tokens) # Use original token count before extensive filtering for density base
    keyword_density = {}
    if total_words > 0:
        for keyword, count in top_keywords_freq.items():
             # Count occurrences in the original cleaned text (case-insensitive) for density
             occurrences = len(re.findall(r'\b' + re.escape(keyword) + r'\b', cleaned_text, re.IGNORECASE))
             density = round((occurrences / total_words) * 100, 2) if total_words > 0 else 0
             keyword_density[keyword] = f"{density}%"


    # --- Attempt TF-IDF (Optional complement/alternative) ---
    # TF-IDF needs multiple documents (sentences) to be meaningful here
    top_keywords_tfidf = {}
    try:
        sentences = sent_tokenize(text_content) # Use original text for sentence context
        if len(sentences) > 1: # Need more than one sentence for TF-IDF
            vectorizer = TfidfVectorizer(
                stop_words=list(stop_words_set),
                ngram_range=(1, 3), # Consider 1, 2, and 3-word phrases
                max_features=top_n,
                max_df=0.85, # Ignore terms that appear in > 85% of sentences
                min_df=1 # Must appear at least once
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            # Aggregate scores across sentences (e.g., max or sum)
            scores = tfidf_matrix.max(axis=0).toarray().flatten() # Max TF-IDF score for each term across all sentences
            top_keywords_tfidf = {feature_names[i]: round(scores[i], 3) for i in scores.argsort()[-top_n:][::-1] if scores[i] > 0}
    except ValueError as e:
        print(f"TF-IDF calculation warning: {e}") # Often happens with very short/uniform text
        # Fallback to frequency already handled

    # Decide which keywords to prioritize (e.g., TF-IDF if available, else frequency)
    final_top_keywords = top_keywords_tfidf if top_keywords_tfidf else top_keywords_freq
     # Format for output
    formatted_keywords = [{"keyword": kw, "score": score} for kw, score in final_top_keywords.items()]


    # --- Generate Summary (Extractive based on keywords) ---
    sentences = sent_tokenize(text_content)
    sentence_scores = Counter()
    keywords_for_summary = list(final_top_keywords.keys())[:5] # Use top 5 for summary scoring

    for sentence in sentences:
        score = 0
        for keyword in keywords_for_summary:
            if keyword in sentence.lower():
                 # Simple scoring: +1 per keyword match per sentence
                 score += sentence.lower().count(keyword)
                 # Optional: Weight longer keywords more, or TF-IDF scores
        if score > 0:
             sentence_scores[sentence] = score

    # Select top ~3 sentences for summary
    num_summary_sentences = min(3, len(sentence_scores))
    summary_sentences = [sent for sent, score in sentence_scores.most_common(num_summary_sentences)]
    summary = " ".join(summary_sentences) if summary_sentences else "Could not generate summary."

    # --- Dynamic SEO Suggestions ---
    seo_suggestions_list = []
    top_5_keys = list(final_top_keywords.keys())[:5]
    if top_5_keys:
        # Suggest long-tail variations (simple patterns)
        seo_suggestions_list.extend([f"Best {kw} strategies", f"How to use {kw} effectively", f"{kw} benefits"] for kw in top_5_keys[:3])        # Suggest content ideas
        seo_suggestions_list.append(f"Consider writing about '{top_5_keys[0]} vs {top_5_keys[1]}'") # Example comparison post
        seo_suggestions_list.append(f"Create an 'Ultimate Guide to {top_5_keys[0]}'") # Example guide post
    else:
        seo_suggestions_list.append("Identify primary and secondary keywords for this page's topic.")

    guidelines = [
        "Integrate top keywords naturally into the Title, H1, meta description, body text, and image alt text.",
        "Aim for a natural keyword density (typically 1-2% for primary keywords, avoid stuffing).",
        "Use a mix of short-tail and long-tail keywords.",
        "Focus on user intent and providing valuable content around these topics.",
        "Internal linking: Link relevant keywords to other related pages on your site."
    ]

    return {
        "top_keywords": formatted_keywords, # List of {"keyword": kw, "score": score}
        "keyword_density": keyword_density, # Dict of {keyword: "density%"}
        "summary": summary,
        "gist": f"The page appears to focus on: {', '.join(top_5_keys)}." if top_5_keys else "Could not determine main topics.",
        "seo_suggestions": {
            "suggested_keywords_ideas": seo_suggestions_list,
            "guidelines": guidelines
        },
        "error": None # Clear previous error if successful
    }


# 9. Anchor Tag Analysis
def analyze_anchor_tags(page_content):
    """
    Analyzes anchor text for generic phrases and counts internal/external links.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    anchor_tags = soup.find_all('a', href=True)
    total_anchors = 0
    generic_anchors = []
    internal_links = 0
    external_links = 0
    missing_text_anchors = 0

    # Get base URL to distinguish internal/external
    base_url_tag = soup.find('base', href=True)
    base_url = base_url_tag['href'] if base_url_tag else None
    # If no <base>, try to infer from common tags (less reliable)
    # This part requires the *actual* URL of the page being analyzed,
    # which isn't passed to this function currently. Needs refactoring
    # to pass the URL or derive it if needed for accurate internal/external split.
    # For now, we'll focus on anchor text quality.

    generic_texts = {"click here", "read more", "learn more", "here", "link", "download", "more info"}

    for anchor in anchor_tags:
         href = anchor.get('href', '')
         text = anchor.get_text(strip=True).lower()

         # Skip mailto, tel, javascript links etc.
         if not href or urlparse(href).scheme in ['mailto', 'tel', 'javascript']:
             continue

         total_anchors += 1

         if not text:
             missing_text_anchors += 1
             generic_anchors.append({"href": href, "text": "[NO TEXT]", "issue": "Anchor has no text"})
         elif text in generic_texts:
             generic_anchors.append({"href": href, "text": anchor.get_text(strip=True), "issue": "Generic anchor text"})

         # Basic internal/external check (needs improvement with actual page URL)
         parsed_href = urlparse(urljoin(base_url or '', href)) # Needs base_url!
         if parsed_href.netloc and base_url and parsed_href.netloc != urlparse(base_url).netloc:
             external_links += 1
         elif not parsed_href.netloc or (base_url and parsed_href.netloc == urlparse(base_url).netloc):
             internal_links += 1
         else: # Could be relative link without base url known
             internal_links += 1 # Assume internal if domain unclear


    suggestions = []
    if generic_anchors:
        suggestions.append(f"Found {len(generic_anchors)} links with generic or missing anchor text (e.g., 'click here', empty). Replace these with descriptive text that includes relevant keywords about the linked page's topic.")
    else:
        suggestions.append("Anchor texts generally appear descriptive. Ensure they accurately reflect the linked content.")

    suggestions.append("Good anchor text improves user experience and helps search engines understand the context of the linked page.")
    suggestions.append("Balance internal and external links. Internal links help site navigation and spread link equity. Relevant external links can provide value to users.")
    # Add suggestion about passing page URL for accurate internal/external count if needed.


    return {
        "total_anchors": total_anchors,
        "generic_or_missing_anchors": generic_anchors, # List of dicts with details
        "internal_link_count": internal_links, # Note: Accuracy depends on base URL knowledge
        "external_link_count": external_links, # Note: Accuracy depends on base URL knowledge
        "suggestion": suggestions
    }


# 10. URL Structure Optimization
def analyze_url_structure(url):
    """
    Analyzes the structure of the given URL path.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path
    issues = []
    suggestions = []

    # Check length (recommendation varies, ~75 chars is a reasonable guideline)
    max_len = 75
    if len(url) > max_len: # Check full URL length as well
        issues.append(f"URL length ({len(url)} chars) is quite long (over {max_len}). Shorter URLs are often preferred.")

    # Check path specifics
    if path and path != '/': # Ignore root path
        # Check for characters to avoid
        if any(c in path for c in ['_', ' ', '%20']):
             issues.append("URL path contains underscores, spaces, or encoded spaces ('%20').")
             suggestions.append("Use hyphens (-) instead of underscores (_) or spaces to separate words in URLs for better readability and SEO.")

        # Check for excessive parameters (simple check)
        if len(parsed_url.query) > 50: # Arbitrary threshold
             issues.append("URL has many parameters, which can sometimes make it less user-friendly or crawlable.")
             suggestions.append("Consider using shorter, descriptive URL paths (rewriting) instead of relying heavily on parameters where possible.")

        # Check depth (simple slash count)
        depth = path.strip('/').count('/')
        if depth > 4: # Arbitrary depth threshold
             issues.append(f"URL path depth ({depth} levels) seems high. Deeply nested pages might be harder for users and crawlers to find.")
             suggestions.append("Aim for a flatter site structure where practical. Ensure important pages are reachable within a few clicks from the homepage.")

        # Check for file extensions (often better to omit)
        if re.search(r'\.(html|htm|php|asp|aspx)$', path):
             issues.append("URL includes a file extension (e.g., .html, .php).")
             suggestions.append("Consider configuring your server to remove file extensions from URLs for a cleaner look.")

    if not issues:
        suggestions.append("URL structure appears reasonable. Ensure it is descriptive, uses hyphens for word separation, and avoids excessive length or parameters.")
    else:
        suggestions.append("Review the URL structure based on the identified issues. Aim for clean, logical, and user-friendly URLs.")

    return {
        "url": url,
        "path": path,
        "query_params": parsed_url.query,
        "issues": issues,
        "suggestions": suggestions
        }


# 11. Robots.txt Validation
def validate_robots_txt(url):
    """
    Checks for the presence and accessibility of robots.txt.
    """
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    robots_url = urljoin(base_url, "/robots.txt")
    status = "Unknown"
    content = None
    error = None
    suggestions = []

    try:
        response = requests.get(robots_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            status = "Exists and Accessible"
            content = response.text
            # Basic checks within the content
            if "Disallow: /" in content.replace(" ", ""): # Basic check for disallowing everything
                 suggestions.append("Warning: Found 'Disallow: /' which blocks all standard crawlers. Ensure this is intended.")
            if "User-agent: *" not in content:
                 suggestions.append("Consider adding a 'User-agent: *' section to apply rules to all standard crawlers, unless specific targeting is needed.")
            if "Sitemap:" not in content:
                 suggestions.append("Consider adding a 'Sitemap:' directive pointing to your XML sitemap location for better discovery.")
            else:
                 suggestions.append("Robots.txt exists and contains rules. Review its directives (Allow, Disallow, Crawl-delay) to ensure they align with your crawling strategy.")

        elif response.status_code == 404:
            status = "Missing (404 Not Found)"
            suggestions.append("Create a robots.txt file at the root of your domain to guide search engine crawlers. Even an empty file or one allowing all access (`User-agent: *\\nAllow: /`) is better than none.")
        else:
            status = f"Exists but Inaccessible (Status: {response.status_code})"
            suggestions.append(f"Robots.txt exists but returned status {response.status_code}. Ensure it's publicly accessible with a 200 OK status.")
            error = f"Received status code {response.status_code}"

    except requests.exceptions.Timeout:
         error = f"Timeout accessing {robots_url}"
         status = "Error (Timeout)"
         suggestions.append(f"Could not access robots.txt due to a timeout. Check server responsiveness.")
    except requests.exceptions.RequestException as e:
        error = f"Failed to check robots.txt: {str(e)}"
        status = "Error (Request Failed)"
        suggestions.append(f"Could not access robots.txt due to a network error ({error}). Verify the URL and server status.")
    except Exception as e:
        error = f"An unexpected error occurred: {str(e)}"
        status = "Error (Unexpected)"

    return {
        "robots_url": robots_url,
        "status": status,
        "content": content, # Be careful displaying full content in UI if sensitive
        "error": error,
        "suggestions": suggestions
        }


# 12. XML Sitemap Validation
def validate_sitemap(url):
    """
    Checks for common XML sitemap locations (sitemap.xml).
    """
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    # Common sitemap locations
    sitemap_locations = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"), # Common for index files
    ]
    found_sitemap_url = None
    status = "Not Found"
    error = None
    suggestions = []

    for sitemap_url in sitemap_locations:
        try:
            response = requests.head(sitemap_url, headers=HEADERS, timeout=REQUEST_TIMEOUT) # HEAD request first
            if response.status_code == 200:
                status = "Exists and Accessible"
                found_sitemap_url = sitemap_url
                suggestions.append(f"Sitemap found at {found_sitemap_url}. Ensure it's valid XML, up-to-date, and submitted to Google Search Console and Bing Webmaster Tools.")
                suggestions.append("Regularly check the sitemap for errors (e.g., using Search Console's report).")
                break # Found one, stop checking
            elif response.status_code == 404:
                continue # Try next location
            else:
                 status = f"Error (Status: {response.status_code})"
                 error = f"Received status code {response.status_code} for {sitemap_url}"
                 suggestions.append(f"Checked {sitemap_url} but received status {response.status_code}. Ensure sitemaps are accessible.")
                 # Don't break, maybe another location works

        except requests.exceptions.Timeout:
            error = f"Timeout accessing {sitemap_url}"
            status = "Error (Timeout)"
            suggestions.append(f"Could not access sitemap at {sitemap_url} due to a timeout.")
            # Don't break, maybe another location works
        except requests.exceptions.RequestException as e:
            error = f"Failed to check sitemap at {sitemap_url}: {str(e)}"
            status = "Error (Request Failed)"
            suggestions.append(f"Could not access sitemap at {sitemap_url} due to a network error.")
            # Don't break, maybe another location works
        except Exception as e:
             error = f"An unexpected error occurred checking {sitemap_url}: {str(e)}"
             status = "Error (Unexpected)"

    if not found_sitemap_url and status != "Error (Timeout)" and status != "Error (Request Failed)" and status != "Error (Unexpected)":
        status = "Missing"
        suggestions.append("No common sitemap (sitemap.xml, sitemap_index.xml) found at the root.")
        suggestions.append("Create an XML sitemap to help search engines discover all important pages on your site. Include it in your robots.txt file and submit it to search consoles.")

    return {
        "checked_locations": sitemap_locations,
        "status": status,
        "found_sitemap_url": found_sitemap_url,
        "error": error,
        "suggestions": suggestions
        }


# 13. Blog Optimization (Reworked without scraping)
def generate_title_suggestions(current_title, h1_texts, keywords):
    """Generates title suggestions based on content and common patterns."""
    suggestions = []
    top_keywords = [kw['keyword'] for kw in keywords[:3]] # Use top 3 keywords

    # Suggestion 1: Enhance current title
    if current_title and top_keywords:
        suggestions.append(f"{current_title} | Key Insights on {top_keywords[0]}")

    # Suggestion 2: Use H1 + Keyword
    if h1_texts and top_keywords:
        suggestions.append(f"{h1_texts[0]}: A Guide to {top_keywords[0]}")

    # Suggestion 3: Common Patterns
    if top_keywords:
        suggestions.append(f"Ultimate Guide to {top_keywords[0]}")
        suggestions.append(f"5 Tips for Effective {top_keywords[0]}")
        if len(top_keywords) > 1:
             suggestions.append(f"Understanding {top_keywords[0]} and {top_keywords[1]}")

    # Ensure uniqueness and relevance
    unique_suggestions = list(dict.fromkeys(suggestions)) # Remove duplicates
    # Basic filtering (optional)
    unique_suggestions = [s for s in unique_suggestions if len(s) < 70] # Keep reasonable length

    return unique_suggestions[:5] # Return top 5


def analyze_blog_optimization(page_content, url):
    """
    Analyzes blog post elements like title, H1, and suggests improvements
    based on content keywords and patterns. No external scraping.
    Relies on keyword analysis results from 'analyze_keyword_summary'.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    title_tag = soup.find('title')
    current_title = title_tag.get_text(strip=True) if title_tag else None
    h1_tags = soup.find_all('h1')
    h1_texts = [h1.get_text(strip=True) for h1 in h1_tags if h1.get_text(strip=True)]

    # --- This function now implicitly relies on keywords found by ---
    # --- analyze_keyword_summary. For a standalone version, keywords ---
    # --- would need to be extracted here. We assume keywords are available ---
    # --- from the main analysis dictionary later. ---
    # --- As a fallback, do a mini-extraction here if needed, but ideally use the main result ---
    keyword_info = analyze_keyword_summary(page_content) # Re-run (inefficient) or get from main results
    top_keywords = keyword_info.get("top_keywords", [])

    suggested_titles = []
    if current_title or h1_texts:
        suggested_titles = generate_title_suggestions(current_title, h1_texts, top_keywords)

    suggestions = []
    if not current_title:
         suggestions.append("Page is missing a <title> tag. Add a compelling title reflecting the content.")
    if not h1_texts:
         suggestions.append("Page is missing an H1 tag. Add a primary H1 heading that matches the main topic.")
    elif len(h1_texts) > 1:
         suggestions.append("Multiple H1 tags found. Use only one H1 for the main title of the blog post.")

    if current_title and h1_texts and current_title.lower() != h1_texts[0].lower():
        suggestions.append("The Title tag and H1 tag differ. Ensure both accurately represent the content and include primary keywords, though they don't have to be identical.")

    if suggested_titles:
        suggestions.append("Consider the suggested titles, which incorporate keywords found in the content and follow common engaging patterns.")
    else:
        suggestions.append("Focus on crafting a clear, keyword-rich title and H1 tag that accurately reflect the blog post's main topic.")

    suggestions.append("Ensure your blog post content is well-structured with subheadings (H2, H3), provides value, and naturally incorporates relevant keywords.")


    return {
        "current_title": current_title,
        "h1_tags": h1_texts,
        "suggested_titles": suggested_titles, # Based on content keywords/patterns
        # "competitor_titles": [], # Removed - unreliable without search/scraping
        # "google_suggested_titles": [], # Removed - unreliable without search/scraping
        "suggestions": suggestions
    }

# 14. HTML Structure Analysis (New)
def analyze_html_structure(page_content):
    """
    Analyzes the use of semantic HTML5 elements and heading structure logic.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    issues = []
    suggestions = []
    semantic_tags_found = []

    # Check for key HTML5 semantic elements
    semantic_tags = ['header', 'nav', 'main', 'article', 'aside', 'footer', 'section']
    for tag_name in semantic_tags:
        if soup.find(tag_name):
            semantic_tags_found.append(tag_name)

    if not semantic_tags_found:
        suggestions.append("Consider using HTML5 semantic tags like <header>, <nav>, <main>, <article>, <aside>, <footer> to improve document structure and accessibility.")
    elif 'main' not in semantic_tags_found:
        suggestions.append("Consider wrapping the main content of the page within a <main> tag.")
    else:
         suggestions.append(f"Found semantic tags: {', '.join(semantic_tags_found)}. Using these tags helps define the structure for assistive technologies and search engines.")

    # Heading structure check (already partially done in analyze_on_page_optimization, refined here)
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    last_level = 0
    h1_found = False
    for heading in headings:
         try:
            level = int(heading.name[1])
            if level == 1: h1_found = True
            if level > last_level + 1:
                 issue_text = f"Heading level skipped: Found <{heading.name}> after <h{last_level}>."
                 if issue_text not in issues: # Avoid duplicate messages
                      issues.append(issue_text)
            last_level = level
         except (ValueError, IndexError):
             continue # Ignore malformed heading tags like <h7>

    if not h1_found:
         if "Missing H1 tag." not in issues: # Avoid duplicate messages
             issues.append("Missing H1 tag.")

    if issues:
        suggestions.append("Review heading hierarchy (H1-H6). Headings should form a logical outline without skipping levels (e.g., don't jump from H2 to H4). Use only one H1.")
    else:
        suggestions.append("Heading structure appears logical.")

    # Basic check for alt text on images
    images = soup.find_all('img')
    missing_alt_count = 0
    for img in images:
        alt_text = img.get('alt')
        if alt_text is None: # alt attribute missing entirely
            missing_alt_count += 1
        elif not alt_text.strip() and not img.has_attr('role') == 'presentation': # alt is empty, and not decorative
            # Check if it's likely decorative based on heuristics (e.g. spacer.gif) - simplistic
            src = img.get('src', '').lower()
            if 'spacer' not in src and 'transparent' not in src:
                 missing_alt_count += 1


    if missing_alt_count > 0:
         issues.append(f"Found {missing_alt_count} images missing descriptive alt text.")
         suggestions.append("Add descriptive alt text to all meaningful images. For purely decorative images, use an empty alt attribute (alt=\"\"). Alt text improves accessibility and SEO.")

    return {
        "semantic_tags_found": semantic_tags_found,
        "heading_structure_issues": issues,
        "images_missing_alt": missing_alt_count,
        "suggestions": suggestions
    }

# 15. DA/PA/Spam Score Placeholder (New)
def analyze_da_pa_spam(url):
    """
    Placeholder function explaining that DA/PA/Spam Score require external tools/APIs.
    """
    explanation = (
        "Domain Authority (DA), Page Authority (PA), and Spam Score are metrics developed by Moz "
        "(or similar metrics by other providers like SEMrush). Accurate calculation requires access "
        "to their extensive link indexes and proprietary algorithms."
    )
    suggestion = (
        "To check these scores, use dedicated SEO tools like Moz Link Explorer, SEMrush, Ahrefs, etc. "
        "These metrics are not directly provided by search engines and cannot be reliably calculated "
        "without these external services/APIs (which often require subscriptions)."
    )

    return {
        "Domain_Authority": "N/A (Requires External Tool/API)",
        "Page_Authority": "N/A (Requires External Tool/API)",
        "Spam_Score": "N/A (Requires External Tool/API)",
        "explanation": explanation,
        "suggestion": suggestion
    }

