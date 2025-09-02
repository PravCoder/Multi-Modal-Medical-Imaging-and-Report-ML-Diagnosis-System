from django.core.management.base import BaseCommand
from ml.pipelines.training_pipeline import training_tests


class Command(BaseCommand):
    help = "Run training pipeline"

    def handle(self, *args, **options):
        print("[CRON] Running training pipeline...")
        training_tests()
        print("[CRON] Done training pipeline...")


"""
------TEST RUN IT MANUALLY TRAINING PIPELINE FILE--------:
cd "/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/backend"
"/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/venv/bin/python" manage.py training_pipeline
"""