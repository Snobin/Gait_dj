# videoapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
    path('info/',views.info,name='info'),
    path('home/',views.upload_video,name='upload_video'),
    
]
