from django.core.management.base import BaseCommand
from ml.pipelines.feature_pipeline import feature_pipeline_tests

class Command(BaseCommand):
    help = "Run feature pipeline"

    def handle(self, *args, **options):
        print("[CRON] Running feature pipeline...")
        feature_pipeline_tests()
        print("[CRON] Done feature pipeline...")




"""
We are using system cron to automate feature & training pipelines. They run at 2:30 & 2:40 am. 


notes:

open cron tab:
crontab -e


find your project directory if your already in it: /Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System
echo $PWD



paste into cron tab, based on path & using virtual environment python version, with the comments. 
# Run feature pipeline daily at 2:30 AM
30 2 * * * cd "/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/backend" && "/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/venv/bin/python" manage.py feature_pipeline >> /tmp/feature_pipeline.log 2>&1

# Run training pipeline daily at 2:40 AM
40 2 * * * cd "/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/backend" && "/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/venv/bin/python" manage.py training_pipeline >> /tmp/training_pipeline.log 2>&1


once pasted, do this:
Since you're likely in vim editor, do this:
-Press ESC key (to make sure you're in command mode)
-Type :wq (this means "write and quit")
-Press ENTER


to check run:
crontab -l

should see something like:
what you pased



------TEST RUN IT MANUALLY FEATURE PIPELINE FILE--------:
cd "/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/backend"
"/Users/pravachanpatra/Documents/PYTHON/AI_ML_DL/Multi-Modal-Medical-Imaging-and-Report-ML-Diagnosis-System/venv/bin/python" manage.py feature_pipeline





delete all existing cronjobs:
crontab -r

"""