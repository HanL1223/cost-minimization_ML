from zenml import Model,pipeline,step
from steps.data_ingestion_steps import data_ingestion_step

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)

def ml_pipeline ():
    """
    Defining ML Pipeline
    """
    #Data Ingestion
    