from django.urls import path
from . import views
from .views import start_analysis, get_analysis_result
urlpatterns = [
    path('api/analyze/', start_analysis, name='start_analysis'),
    path('api/result/<str:task_id>/', get_analysis_result, name='get_analysis_result'),
]
