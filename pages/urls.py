# pages/urls.py
from django.urls import path
from .views import homePageView, aboutPageView, contactPageView

urlpatterns = [
    path('', homePageView, name='home'),
    path('about', aboutPageView, name='about'),
    path('contact', contactPageView, name='contact')
]
