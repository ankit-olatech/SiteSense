
# Create your models here.
from django.db import models

class Webpage(models.Model):
    url = models.URLField(unique=True)
    title = models.CharField(max_length=255)
    meta_description = models.TextField()
    keywords = models.TextField()
    content = models.TextField()

    def __str__(self):
        return self.url
