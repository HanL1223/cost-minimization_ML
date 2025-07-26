from steps.data_evaluator_steps import model_evaluator_step
from steps.data_ingestion_steps import data_ingestion_step
from steps.data_splitting_steps import data_spilting_step
from steps.missing_value_handling_steps import missing_value_imputation
from steps.model_building_steps import model_building_step
from steps.model_selection_steps import model_selection_steps
from steps.model_tunning_steps import model_tunning_steps
import os


from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="cost_predictor"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    file_path = '/Users/hanli/cost-minimization_ML/data/Training_raw/Train.csv'
    raw_data = data_ingestion_step(file_path)

    # Handling Missing Values Step
    filled_data = missing_value_imputation(raw_data,strategy='mean')

    # Feature Engineering Step (phase 2)

    # Outlier Detection Step (phase 2)

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_spilting_step(filled_data, target_column="Target")

    #Model Selection Step
    best_model_name,best_model,model_path = model_selection_steps(X_train,y_train)

    #Best Model Tunning
    best_model_path = model_tunning_steps(best_model_name,best_model,X_train,y_train)

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
