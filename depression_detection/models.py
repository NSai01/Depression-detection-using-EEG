from django.db import models

class UserProfile(models.Model):
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=200)
    email = models.EmailField(blank=True)

    def __str__(self):
        return self.username

    class Meta:
        db_table = 'user_profile'


class DatasetRecord(models.Model):
    filename = models.CharField(max_length=255)
    total_records = models.IntegerField(default=0)
    normal_count = models.IntegerField(default=0)
    depressed_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename

    class Meta:
        db_table = 'dataset_record'


class AlgorithmResult(models.Model):
    ALGORITHM_CHOICES = [('SVM', 'SVM'), ('CNN', 'CNN')]
    algorithm = models.CharField(max_length=10, choices=ALGORITHM_CHOICES)
    accuracy = models.FloatField(default=0)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    fscore = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm} - Accuracy: {self.accuracy:.2f}%"

    class Meta:
        db_table = 'algorithm_result'


class PredictionHistory(models.Model):
    RESULT_CHOICES = [('Normal', 'Normal'), ('Depressed', 'Depressed')]
    signal_filename = models.CharField(max_length=255)
    prediction_result = models.CharField(max_length=20, choices=RESULT_CHOICES)
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.signal_filename} - {self.prediction_result}"

    class Meta:
        db_table = 'prediction_history'
