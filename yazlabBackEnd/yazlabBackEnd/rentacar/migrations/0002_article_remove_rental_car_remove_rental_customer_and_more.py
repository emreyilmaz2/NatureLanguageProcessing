# Generated by Django 5.0.6 on 2024-05-16 10:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rentacar', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Article',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
            ],
        ),
        migrations.RemoveField(
            model_name='rental',
            name='car',
        ),
        migrations.RemoveField(
            model_name='rental',
            name='customer',
        ),
        migrations.DeleteModel(
            name='Vehicle',
        ),
        migrations.DeleteModel(
            name='Rental',
        ),
    ]