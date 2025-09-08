from django.core.management.base import BaseCommand
from ml.pipelines.feature_pipeline import feature_pipeline_tests    # import the function that runs the feature pipeline


class Command(BaseCommand):
    help = "runs the daily feature pipeline"  # description of the command

    def handle(self, *args, **options):

        self.stdout.write(self.style.WARNING("[FEATURE-PIPE] started running the daily feature pipeline..."))
        
        try:

            # call the feature-pipeline function
            feature_pipeline_tests()           
            
            self.stdout.write(self.style.SUCCESS("[FEATURE-PIPE] done running the feature pipeline..."))
        
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"[FEATURE-PIPE] error running pipeline: {e}"))
            raise e
        

"""
To test piplines locally:

cd backend:
run: python3 manage.py run_daily_feature_pipeline
run: python3 manage.py run_daily_training_pipeline


"""