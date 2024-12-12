from django.http import JsonResponse
from django.shortcuts import render
from .models import *
import requests
from bs4 import BeautifulSoup

def index(request):
    return render(request, 'test.html')

def analyze_page(request):
    url = request.GET.get('url', '')
    
    if not url:
        return JsonResponse({"error": "URL parameter is required"}, status=400)
    
    # Fetch the page
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title
        title = soup.title.string if soup.title else "No title found"
        
        # Perform further SEO analysis here (e.g., H1 tag check, meta tag check)
        
        analysis_results = {
            "title": title,
            "meta_description": soup.find('meta', {'name': 'description'}),
            # Add more analysis results here
        }
        
        # Save the results to the database (optional)
        webpage = Webpage(url=url, title=title, meta_description=analysis_results["meta_description"])
        webpage.save()
        
        return JsonResponse(analysis_results)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)
