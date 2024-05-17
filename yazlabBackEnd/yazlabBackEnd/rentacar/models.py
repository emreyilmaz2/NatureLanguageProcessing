from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    age = models.IntegerField(blank=True, null=True)
    gender = models.CharField(max_length=10, blank=True, null=True)
    interest = models.CharField(max_length=100, blank=True, null=True)
    is_logged_in = models.BooleanField(default=False)  # Kullanıcı oturum durumu

    def __str__(self):
        return self.username

class Article(models.Model):
    text = models.TextField()
    keywords = models.TextField(max_length=50, default="non-keyword")
    heading = models.TextField(max_length=50, default="non-titled")
    published_date = models.DateTimeField(null=True)  # Yayınlanma tarihi yoksa None yapabilirsiniz
    
    def __str__(self):
        return f"{self.heading}"
