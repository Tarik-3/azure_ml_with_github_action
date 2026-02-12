import os
from pathlib import Path

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core import Dataset
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(Path(__file__).with_name(".env"))

# Connect to workspace using secrets (environment variables or .env)
ws = Workspace(
    subscription_id=os.environ["AZUREML_SUBSCRIPTION_ID"],
    resource_group=os.environ["AZUREML_RESOURCE_GROUP"],
    workspace_name=os.environ["AZUREML_WORKSPACE_NAME"]
)

# Define the AzureML environment used by all steps
env = Environment.from_conda_specification(name="ml-env", file_path="environment.yml")

# Shared outputs for the pipeline
prepared_data = PipelineData("prepared_data", datastore=ws.get_default_datastore())
model_output = PipelineData("model_output", datastore=ws.get_default_datastore())

data_source = "azureml"
azureml_data_uri = os.environ.get(
    "AZUREML_DATA_URI",
    (
        "azureml://subscriptions/3c903801-0878-49d9-9d2c-3ed7f0e0ad1c/"
        "resourcegroups/RG_JIT02/workspaces/data-factory/datastores/"
        "workspaceblobstore/paths/UI/2026-02-12_094456_UTC/sample_data.csv"
    ),
)

repo_url = "https://github.com/Tarik-3/azure_ml_with_github_action.git"
repo_ref = "dev"
repo_path = "Amsterdam subsidy assignment/my_work/data.csv"

raw_data_input = None
download_step = None

def _parse_azureml_datastore_uri(uri: str) -> tuple:
    marker = "/datastores/"
    paths_marker = "/paths/"
    if marker not in uri or paths_marker not in uri:
        raise ValueError(f"Unsupported AzureML datastore URI: {uri}")
    datastore_part = uri.split(marker, 1)[1]
    datastore_name = datastore_part.split(paths_marker, 1)[0]
    path = datastore_part.split(paths_marker, 1)[1]
    return datastore_name, path


if data_source == "azureml":
    datastore_name, datastore_path = _parse_azureml_datastore_uri(azureml_data_uri)
    datastore = ws.datastores[datastore_name]
    raw_data_input = Dataset.File.from_files(
        path=(datastore, datastore_path)
    ).as_named_input("raw_data").as_download()
elif repo_url == "local":
    # Upload local sample data so the remote compute can access it
    datastore = ws.get_default_datastore()
    local_file = Path(__file__).resolve().parent / repo_path
    datastore.upload_files(
        files=[str(local_file)],
        target_path="inputs",
        overwrite=True,
    )
    raw_data_input = Dataset.File.from_files(
        path=(datastore, f"inputs/{local_file.name}")
    ).as_named_input("raw_data")
else:
    # Download data from GitHub during the pipeline run
    raw_data_output = PipelineData("raw_data", datastore=ws.get_default_datastore())
    raw_data_input = raw_data_output
    download_config = ScriptRunConfig(source_directory=".", environment=env)
    download_config.environment_variables = {
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", "")
    }

    # Step 1: Download data
    download_step = PythonScriptStep(
        name="Download Data",
        script_name="download_data.py",
        arguments=[
            "--repo",
            repo_url,
            "--ref",
            repo_ref,
            "--path",
            repo_path,
            "--output",
            raw_data_output,
        ],
        outputs=[raw_data_output],
        compute_target="pipe-action",
        source_directory=".",
        runconfig=download_config,
    )

# Step 2: Prep
prep_step = PythonScriptStep(
    name="Prep Data",
    script_name="prep.py",
    arguments=["--input", raw_data_input, "--output", prepared_data],
    inputs=[raw_data_input],
    outputs=[prepared_data],
    compute_target="pipe-action",   # Make sure this compute cluster exists in your workspace
    source_directory=".",
    runconfig=ScriptRunConfig(source_directory=".", environment=env)
)

# Step 3: Train
train_step = PythonScriptStep(
    name="Train Model",
    script_name="train.py",
    arguments=["--input", prepared_data, "--model_output", model_output],
    inputs=[prepared_data],
    outputs=[model_output],
    compute_target="pipe-action",
    source_directory=".",
    runconfig=ScriptRunConfig(source_directory=".", environment=env)
)

# Step 4: Test
test_step = PythonScriptStep(
    name="Test Model",
    script_name="test.py",
    arguments=["--input", prepared_data, "--model_path", model_output],
    inputs=[prepared_data, model_output],
    compute_target="pipe-action",
    source_directory=".",
    runconfig=ScriptRunConfig(source_directory=".", environment=env)
)

# Build pipeline
steps = [prep_step, train_step, test_step]
if download_step is not None:
    steps.insert(0, download_step)

pipeline = Pipeline(workspace=ws, steps=steps)

# Submit experiment
exp = Experiment(workspace=ws, name="diabetes-ml-pipeline")
exp.submit(pipeline)