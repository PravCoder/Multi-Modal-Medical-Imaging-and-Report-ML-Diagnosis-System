from django.core.management.base import BaseCommand
from ml.pipelines.training_pipeline import training_tests           # import the function that runs the 


class Command(BaseCommand):
    help = "runs the daily training pipeline"  # description of the command

    def handle(self, *args, **options):

        self.stdout.write(self.style.WARNING("[TRAINING-PIPE] started running the daily training pipeline..."))
        
        try:

            # call the training-pipeline function
            training_tests()           
            
            self.stdout.write(self.style.SUCCESS("[TRAINING-PIPE] done running the training pipeline..."))
        
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"[TRAINING-PIPE] error running pipeline: {e}"))
            raise e
        


