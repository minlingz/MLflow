# Databricks notebook source
# MAGIC %md # Training machine learning models on tabular data: Resume
# MAGIC
# MAGIC It covers the following steps:
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Run a parallel hyperparameter sweep to train machine learning models on the dataset
# MAGIC - Explore the results of the hyperparameter sweep with MLflow
# MAGIC
# MAGIC In this example, I build a model to predict whether the resume owner will receive call back or not based on the resume properties.
# MAGIC
# MAGIC The example uses a dataset from openintro.
# MAGIC
# MAGIC ## Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning.
# MAGIC If you are using Databricks Runtime 7.3 LTS ML, you must update the CloudPickle library. To do that, uncomment and run the `%pip install` command in Cmd 2.

# COMMAND ----------

# This command is only required if you are using a cluster running DBR 7.3 LTS ML.
#!pip install --upgrade cloudpickle
#!pip install mlflow

# COMMAND ----------

import pandas as pd

data = pd.read_csv("/dbfs/FileStore/shared_uploads/mz246@duke.edu/resume.csv")

# COMMAND ----------

data = data.iloc[:, -16:].drop(["firstname"], axis=1)
data.head(5)

# COMMAND ----------

data["race"] = data["race"].apply(lambda x: 1 if x == "white" else 0)
data["resume_quality"] = data["resume_quality"].apply(lambda x: 1 if x == "high" else 0)
data["gender"] = data["gender"].apply(lambda x: 1 if x == "f" else 0)
data.head(10)

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare dataset for training baseline model
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["received_callback"], axis=1)
y = data.received_callback

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(
    X, y, train_size=0.6, random_state=123
)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=123
)

# COMMAND ----------

# MAGIC %md ## Build a baseline model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

# COMMAND ----------

import os

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

mlflow.sklearn.autolog()

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1).
# The following code creates a wrapper function, SklearnModelWrapper, that uses
# the predict_proba method to return the probability that the observation belongs to each class.


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


# mlflow.start_run creates a new MLflow run to track the performance of this model.
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
# mlflow.create_experiment("mzExperiment")


# COMMAND ----------

with mlflow.start_run(
    run_name="untuned_random_forest",
    experiment_id=mlflow.get_experiment_by_name("mzExperiment").experiment_id,
):
    n_estimators = 10
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=np.random.RandomState(123)
    )
    model.fit(X_train, y_train)

    # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_param("n_estimators", n_estimators)
    # Use the area under the ROC curve as a metric.
    mlflow.log_metric("auc", auc_score)
    wrappedModel = SklearnModelWrapper(model)
    # Log the model with a signature that defines the schema of the model's inputs and outputs.
    # When the model is deployed, this signature will be used to validate inputs.
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

    # MLflow contains utilities to create a conda environment used to serve models.
    # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__),
            "scikit-learn=={}".format(sklearn.__version__),
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        "random_forest_model",
        python_model=wrappedModel,
        conda_env=conda_env,
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md Examine the learned feature importances output by the model as a sanity-check.

# COMMAND ----------

feature_importances = pd.DataFrame(
    model.feature_importances_, index=X_train.columns.tolist(), columns=["importance"]
)
feature_importances.sort_values("importance", ascending=False)

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Set the value of 'run_name'
run_name = "untuned_random_forest"

# Retrieve the run ID for the run
# Assumes that you have already set the experiment ID and 'run_name' parameter
search_results = mlflow.search_runs(
    experiment_ids=mlflow.get_experiment_by_name("mzExperiment").experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    filter_string=f"tags.mlflow.runName = '{run_name}'",
)
run_id = search_results.loc[search_results["tags.mlflow.runName"] == run_name][
    "run_id"
].iloc[0]

# Retrieve the AUC metric value from the run
run = mlflow.get_run(run_id)
auc = run.data.metrics["auc"]
print(auc)

# COMMAND ----------

# MAGIC %md #### Register the model in MLflow Model Registry
# MAGIC
# MAGIC By registering this model in Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI. See "Create or register a model using the UI" ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)).

# COMMAND ----------

model_name = "resume"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
)

# COMMAND ----------

# MAGIC %md The Models page now shows the model version in stage "Production".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f"AUC: {roc_auc_score(y_test, model.predict(X_test))}")

# COMMAND ----------

# MAGIC %md ##Experiment with a new model
# MAGIC
# MAGIC The random forest model performed well even without hyperparameter tuning.
# MAGIC
# MAGIC The following code uses the xgboost library to train a more accurate model. It runs a parallel hyperparameter sweep to train multiple
# MAGIC models in parallel, using Hyperopt and SparkTrials. As before, the code tracks the performance of each parameter configuration with MLflow.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "binary:logistic",
    "seed": 123,  # Set a seed for deterministic training
}


def train_model(params):
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    mlflow.xgboost.autolog()
    with mlflow.start_run(nested=True):
        train = xgb.DMatrix(data=X_train, label=y_train)
        validation = xgb.DMatrix(data=X_val, label=y_val)
        # Pass in the validation set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
        # is no longer improving.
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(validation, "validation")],
            early_stopping_rounds=50,
        )
        validation_predictions = booster.predict(validation)
        auc_score = roc_auc_score(y_val, validation_predictions)
        mlflow.log_metric("auc", auc_score)

        signature = infer_signature(X_train, booster.predict(train))
        mlflow.xgboost.log_model(booster, "model", signature=signature)

        # Set the loss to -1*auc_score so fmin maximizes the auc_score
        return {
            "status": STATUS_OK,
            "loss": -1 * auc_score,
            "booster": booster.attributes(),
        }


# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(
    run_name="xgboost_models",
    experiment_id=mlflow.get_experiment_by_name("mzExperiment").experiment_id,
):
    best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=96,
        trials=spark_trials,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use MLflow to view the results
# MAGIC Open up the Experiment Runs sidebar to see the MLflow runs. Click on Date next to the down arrow to display a menu, and select 'auc' to display the runs sorted by the auc metric. The highest auc value is 0.64.
# MAGIC
# MAGIC MLflow tracks the parameters and performance metrics of each run. Click the External Link icon <img src="https://docs.databricks.com/_static/images/icons/external-link.png"/> at the top of the Experiment Runs sidebar to navigate to the MLflow Runs Table.

# COMMAND ----------

# MAGIC %md Now investigate how the hyperparameter choice correlates with AUC. Click the "+" icon to expand the parent run, then select all runs except the parent, and click "Compare". Select the Parallel Coordinates Plot.
# MAGIC
# MAGIC The Parallel Coordinates Plot is useful in understanding the impact of parameters on a metric. You can drag the pink slider bar at the upper right corner of the plot to highlight a subset of AUC values and the corresponding parameter values. The plot below highlights the highest AUC values:
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC
# MAGIC Notice that all of the top performing runs have a low value for reg_lambda and learning_rate.
# MAGIC
# MAGIC You could run another hyperparameter sweep to explore even lower values for these parameters. For simplicity, that step is not included in this example.

# COMMAND ----------

# MAGIC %md
# MAGIC You used MLflow to log the model produced by each hyperparameter configuration. The following code finds the best performing run and saves the model to Model Registry.
# MAGIC

# COMMAND ----------

run = mlflow.search_runs(
    experiment_ids=mlflow.get_experiment_by_name("mzExperiment").experiment_id,
    order_by=["metrics.auc DESC"],
)
print(run["metrics.auc"])
# best_run = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name('mzExperiment').experiment_id,order_by=['metrics.auc DESC']).iloc[0]
# print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### Update the production `resume` model in MLflow Model Registry
# MAGIC
# MAGIC Earlier, you saved the baseline model to Model Registry with the name `resume`. Now that you have a created a more accurate model, update `resume`.

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md Click **Models** in the left sidebar to see that the `resume` model now has two versions.
# MAGIC
# MAGIC The following code promotes the new version to production.

# COMMAND ----------

# Archive the old model version
client.transition_model_version_stage(
    name=model_name, version=model_version.version, stage="Archived"
)

# Promote the new model version to Production
client.transition_model_version_stage(
    name=model_name, version=new_model_version.version, stage="Production"
)

# COMMAND ----------

# MAGIC %md Clients that call load_model now receive the new model.

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f"AUC: {roc_auc_score(y_test, model.predict(X_test))}")

# COMMAND ----------

# MAGIC %md The auc value on the test set for the new model is 0.64. It beat the baseline!
