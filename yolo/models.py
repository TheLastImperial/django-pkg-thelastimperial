from django.db import models

# Create your models here.
class Prediction(models.Model):
    original = models.ImageField(upload_to="original", )
    processed = models.ImageField(upload_to="processed", )
    created_at = models.DateTimeField("Created published")

    def __str__(self):
        return created_at
