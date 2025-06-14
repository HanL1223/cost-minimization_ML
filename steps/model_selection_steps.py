from zenml import step



def model_selection():
    models = []  # Empty list to store all the models

    # Appending models into the list

    models.append(
        ("Logistic Regression", LogisticRegression(solver="newton-cg", random_state=1))
    )
    models.append(("dtree", DecisionTreeClassifier(random_state=1)))
    models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))