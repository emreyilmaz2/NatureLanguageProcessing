# Generated by Django 5.0.6 on 2024-05-16 16:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rentacar', '0003_article_published_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='keywords',
            field=models.TextField(default='non-keyword'),
        ),
    ]