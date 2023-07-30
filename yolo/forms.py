# from django.forms import ModelForm
from django.forms import Form, CharField, ImageField
from .models import Prediction

class PredictionForm(Form):
	original = ImageField()
