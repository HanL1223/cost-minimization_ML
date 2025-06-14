from zenml import Model,pipeline,step
from steps.data_ingestion_steps import data_ingestion_step
from steps.missing_value_handling_steps import missing_value_imputation
from steps.data_splitting_steps import data_spilting_step
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
    file_path = '/Users/hanli/cost-minimization_ML/data/Training_raw/Train.csv'
    raw_data = data_ingestion_step(file_path)

    #Missing Value Handling
    clean_data = missing_value_imputation(raw_data,strategy='median')

    #Data splitting steps
    X_train, X_test, y_train, y_test = data_spilting_step(clean_data,target_column ='Target')

    #Model Selection and validation

    #Model Evaulation

