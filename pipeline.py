from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Connect to workspace
ws = Workspace.from_config()

# Define environment
env = Environment.from_conda_specification(name="ml-env", file_path="environment.yml")

# Shared outputs
prepared_data = PipelineData("prepared_data", datastore=ws.get_default_datastore())
model_output = PipelineData("model_output", datastore=ws.get_default_datastore())

# Step 1: Prep
prep_step = PythonScriptStep(
    name="Prep Data",
    script_name="prep.py",
    arguments=["--output", prepared_data],
    outputs=[prepared_data],
    compute_target="cpu-cluster",
    source_directory=".",
    runconfig=ScriptRunConfig(source_directory=".", environment=env)
)

# Step 2: Train
train_step = PythonScriptStep(
    name="Train Model",
    script_name="train.py",
    arguments=["--input", prepared_data, "--model_output", model_output],
    inputs=[prepared_data],
    outputs=[model_output],
    compute_target="cpu-cluster",
    source_directory=".",
    runconfig=ScriptRunConfig(source_directory=".", environment=env)
)

# Step 3: Test
test_step = PythonScriptStep(
    name="Test Model",
    script_name="test.py",
    arguments=["--input", prepared_data, "--model_path", model_output],
    inputs=[prepared_data, model_output],
    compute_target="cpu-cluster",
    source_directory=".",
    runconfig=ScriptRunConfig(source_directory=".", environment=env)
)

# Build pipeline
pipeline = Pipeline(workspace=ws, steps=[prep_step, train_step, test_step])

# Submit experiment
exp = Experiment(workspace=ws, name="diabetes-ml-pipeline")
exp.submit(pipeline)