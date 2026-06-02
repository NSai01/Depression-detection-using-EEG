from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('features-extraction/', views.features_extraction, name='features_extraction'),
    path('run-svm/', views.run_svm, name='run_svm'),
    path('run-cnn/', views.run_cnn, name='run_cnn'),
    path('predict/', views.predict_depression, name='predict_depression'),
    path('comparison-graph/', views.comparison_graph, name='comparison_graph'),
    path('prediction-history/', views.prediction_history, name='prediction_history'),
]
