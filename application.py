import json
import os
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file

from backorder.config.configuration import Configuration
from backorder.constant import CONFIG_DIR, get_current_time_stamp
from backorder.entity.backorder_predictor import BackorderData, BackorderPredictor
from backorder.io import read_yaml_file, write_yaml_file
from backorder.logger import get_log_dataframe
from backorder.pipeline.pipeline import Pipeline

ROOT_DIR = Path.cwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "backorder"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = ROOT_DIR / CONFIG_DIR / "model.yaml"
LOG_DIR = ROOT_DIR / LOG_FOLDER_NAME
PIPELINE_DIR = ROOT_DIR / PIPELINE_FOLDER_NAME
MODEL_DIR = ROOT_DIR / SAVED_MODELS_DIR_NAME
BACKORDER_DATA_KEY = "backorder_data"
WENT_ON_BACK_ORDER_KEY = "went_on_backorder"

app = application = Flask(__name__)


@app.route("/artifact", defaults={"req_path": "backorder"})
@app.route("/artifact/<path:req_path>")
def render_artifact_dir(req_path):
    os.makedirs("backorder", exist_ok=True)
    abs_path = os.path.join(req_path)

    if not os.path.exists(abs_path):
        return abort(404)

    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ""
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    files = {
        os.path.join(abs_path, file_name): file_name
        for file_name in os.listdir(abs_path)
        if "artifact" in os.path.join(abs_path, file_name)
    }

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path,
    }
    return render_template("files.html", result=result)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/view_experiment_hist", methods=["GET", "POST"])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {"experiment": experiment_df.to_html(classes="table table-striped col-12")}
    return render_template("experiment_history.html", context=context)


@app.route("/train", methods=["GET", "POST"])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(
            classes="table table-striped col-12"
        ),
        "message": message,
    }
    return render_template("train.html", context=context)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = {BACKORDER_DATA_KEY: None, WENT_ON_BACK_ORDER_KEY: None}

    if request.method == "POST":
        backorder_data = BackorderData(**request.form)
        backorder_df = backorder_data.get_backorder_input_data_frame()
        backorder = BackorderPredictor(model_dir=MODEL_DIR)
        went_on_backorder = backorder.predict(X=backorder_df)
        context = {
            BACKORDER_DATA_KEY: backorder_data.get_backorder_data_as_dict(),
            WENT_ON_BACK_ORDER_KEY: went_on_backorder,
        }
        return render_template("predict.html", context=context)

    return render_template("predict.html", context=context)


@app.route("/saved_models", defaults={"req_path": "saved_models"})
@app.route("/saved_models/<path:req_path>")
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    abs_path = os.path.join(req_path)

    if not os.path.exists(abs_path):
        return abort(404)

    if os.path.isfile(abs_path):
        return send_file(abs_path)

    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path,
    }
    return render_template("saved_model_files.html", result=result)


@app.route("/update_model_config", methods=["GET", "POST"])
def update_model_config():
    if request.method == "POST":
        model_config = request.form["new_model_config"]
        model_config = model_config.replace("'", '"')
        model_config = json.loads(model_config)

        write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

    model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
    return render_template("update_model.html", result={"model_config": model_config})


@app.route("/logs", defaults={"req_path": f"{LOG_FOLDER_NAME}"})
@app.route(f"/{LOG_FOLDER_NAME}/<path:req_path>")
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    abs_path = os.path.join(req_path)

    if not os.path.exists(abs_path):
        return abort(404)

    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template("log.html", context=context)

    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path,
    }
    return render_template("log_files.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
