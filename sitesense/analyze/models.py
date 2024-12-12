
# Create your models here.
from django.db import models

class Webpage(models.Model):
    url = models.URLField()
    title = models.CharField(max_length=255)
    meta_description = models.TextField(null=True)
    keywords = models.TextField(null=True)
    content = models.TextField(null=True)

    def __str__(self):
        return self.url
