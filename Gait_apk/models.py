from django.db import models

# Create your models here.
# videoapp/models.py

from django.db import models

class Video(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='videos/')
