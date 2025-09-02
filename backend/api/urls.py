from django.urls import path
from .views import get_items    # import the views
from .views import predict_view, load_random_sample_view


urlpatterns = [
    path("items/", get_items),
    path("predict/", predict_view, name="predict"),
    path("load-sample/", load_random_sample_view, name="load-sample")
]