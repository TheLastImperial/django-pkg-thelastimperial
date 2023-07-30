from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import DetailView
from django.utils import timezone

from .models import Prediction
from .forms import PredictionForm
from .helpers import get_yolo_detection

# Create your views here.

class DetailView(DetailView):
	model = Prediction
	template_name = "detail.html"

def get_yolo_form(request):
	prediction = Prediction()
	if request.method == "POST":
		form = PredictionForm(request.POST or None, request.FILES or None)
		if form.is_valid():
			prediction.created_at = timezone.now()
			prediction.original = request.FILES['original']
			prediction.save()
			img = get_yolo_detection(prediction.original.path)
			prediction.processed.save("processed.jpg", img, save=True)

			return HttpResponseRedirect(reverse("yolo:detail", args=(prediction.id,)))
		return render(request, "create.html", {
			"form": form
		})
	else:
		return render(request, "create.html", {
			"form": PredictionForm()
		})
