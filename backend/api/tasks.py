from celery import shared_task
from django.core.mail import send_mail
from ml.pipelines.feature_pipeline import feature_pipeline_tests
from ml.pipelines.training_pipeline import training_tests


# Put our pipeline acripts into Celery tasks module.

@shared_task
def run_daily_feature_pipeline():
    feature_pipeline_tests()        # feature pipeline func
    print("[CELERY] feature pipeline done running.")
    return "feature pipeline completed"

@shared_task
def run_daily_training_pipeline():
    training_tests()               # training pipeline func
    print("[CELERY] training pipeline done running.")
    return "training pipeline completed"