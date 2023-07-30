from django.urls import path

from .views import get_yolo_form, DetailView

app_name = "yolo"

urlpatterns = [
	path("", get_yolo_form, name="create"),
	path("detail/<int:pk>", DetailView.as_view(), name="detail"),
]
