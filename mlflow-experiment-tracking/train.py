import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("nyc-taxi-experiment")

    

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    metrics={
        "rmse":rmse
    }

    with mlflow.start_run(run_name="training") as run:
        # Log the parameters used for the model fit


        params = rf.get_params()
        mlflow.log_params(params)
        mlflow.log_param("min_samples_split", params["min_samples_split"])

        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)

        # Log an instance of the trained model for later use
        mlflow.sklearn.log_model(
            sk_model=rf, input_example=X_val, artifact_path="rf_green_trip"
        )

if __name__ == '__main__':
    run_train()