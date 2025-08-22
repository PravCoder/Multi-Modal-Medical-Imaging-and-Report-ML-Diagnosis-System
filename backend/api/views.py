from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(["GET"])
def get_items(request):
    return Response([{"name": "Item 1"}, {"name": "Item 2"}])  