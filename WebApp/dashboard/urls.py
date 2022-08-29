from django.urls import path
from . import views
"""
URLS:
    /dashboard
    /dashboard/detail


"""
urlpatterns = [
    path('', views.index),
    path('detail', views.detail)
]