# Generated by Django 5.0.1 on 2025-03-01 10:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('userapp', '0003_rename_created_at_predictionresult_timestamp_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='All_model',
            fields=[
                ('model_id', models.AutoField(primary_key=True, serialize=False)),
                ('model_Name', models.TextField(max_length=500)),
                ('model_Accuracy', models.DecimalField(decimal_places=2, max_digits=5)),
                ('model_Status', models.TextField(max_length=500)),
                ('Classes', models.DecimalField(decimal_places=2, max_digits=5)),
            ],
            options={
                'db_table': 'all_models',
            },
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('Data_id', models.AutoField(primary_key=True, serialize=False)),
                ('Image', models.ImageField(upload_to='media/')),
                ('RGB_Image', models.ImageField(blank=True, null=True, upload_to='media/')),
                ('Segmented_Image', models.ImageField(blank=True, null=True, upload_to='media/')),
            ],
            options={
                'db_table': 'upload',
            },
        ),
    ]
