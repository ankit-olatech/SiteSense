from celery import shared_task
from .analysis_logic import analyze_page_synchronously  

@shared_task(bind=True)
def analyze_url_task(self, url):
    return analyze_page_synchronously(url)
