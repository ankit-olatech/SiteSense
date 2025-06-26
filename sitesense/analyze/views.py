from django.http import JsonResponse
from analyze.tasks import analyze_url_task
from django.urls import reverse

from celery.result import AsyncResult
from django.http import JsonResponse

def get_analysis_result(request, task_id):
    result = AsyncResult(task_id)

    if result.state == 'PENDING':
        return JsonResponse({'status': 'pending'})
    elif result.state == 'STARTED':
        return JsonResponse({'status': 'started'})
    elif result.state == 'FAILURE':
        return JsonResponse({'status': 'failed', 'error': str(result.result)})
    elif result.state == 'SUCCESS':
        return JsonResponse({'status': 'success', 'data': result.result})
    else:
        return JsonResponse({'status': result.state})

def start_analysis(request):
    url = request.GET.get('url')
    if not url:
        return JsonResponse({'error': 'URL parameter is required.'}, status=400)
    
    task = analyze_url_task.delay(url)

    # Generate absolute URL to poll status
    result_url = request.build_absolute_uri(
        reverse('get_analysis_result', args=[task.id])
    )

    return JsonResponse({
        'task_id': task.id,
        'status_check_url': result_url
    })

