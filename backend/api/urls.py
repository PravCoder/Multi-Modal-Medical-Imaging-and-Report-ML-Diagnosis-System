from django.urls import path
from .views import get_items    # import the views

urlpatterns = [
    path("items/", get_items),
]