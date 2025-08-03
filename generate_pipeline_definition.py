
import json
import os
from sagemaker import Session
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline


ROLE_ARN = "arn:aws:iam::714556467435:role/NewAiStack-SageMakerRoleD4FCFA3F-sqVUgWwDd5Qq"

script_path = "preprocessing_script.py"
if not os.path.exists(script_path):
    with open(script_path, "w") as f:
        f.write("# Placeholder preprocessing script\n")

session = Session()
role = ROLE_ARN

# Define pipeline parameters
raw_data_uri    = ParameterString(name="RawDataUri")
proc_train_uri  = ParameterString(name="ProcTrainDataUri")
proc_val_uri    = ParameterString(name="ProcValDataUri")
instance_type   = ParameterString(name="InstanceType")
instance_count  = ParameterInteger(name="InstanceCount")
model_pkg_group = ParameterString(name="ModelPackageGroup")

processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
    command=["python3"],
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    max_runtime_in_seconds=3600,
    sagemaker_session=session
)
preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=processor,
    inputs=[ProcessingInput(source=raw_data_uri, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(output_name="train_data", source="/opt/ml/processing/output")],
    code=script_path
)

# 2) Training step
estimator = Estimator(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-training:4.17.0-py38-cpu-py310-ubuntu20.04",
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    output_path=proc_train_uri,
    hyperparameters={
        "model_name": "yangheng/deberta-v3-base-absa-v1.1",
        "epochs": 3,
        "train_batch_size": 16,
        "validation_split": 0.1
    }
)
train_inputs = {
    "train": TrainingInput(
        s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
        content_type="application/json"
    ),
    "validation": TrainingInput(
        s3_data=proc_val_uri,
        content_type="application/json"
    )
}
train_step = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs=train_inputs
)

register_step = RegisterModel(
    name="Register",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.t2.medium"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_pkg_group
)

# 4) Assemble the pipeline
pipeline = Pipeline(
    name="SentimentAnalysisPipeline",
    parameters=[raw_data_uri, proc_train_uri, proc_val_uri, instance_type, instance_count, model_pkg_group],
    steps=[preprocess_step, train_step, register_step],
    sagemaker_session=session
)

# 5) Generate and save the JSON definition
full_definition = pipeline.definition()
# Ensure full_definition is a dict
if not isinstance(full_definition, dict):
    # Decode bytes if necessary
    if isinstance(full_definition, (bytes, bytearray)):
        full_definition = full_definition.decode('utf-8')
    full_definition = json.loads(full_definition)

# Build the service-expected body
service_body = {
    "Version": "2020-12-01",
    "Metadata": {"PipelineName": pipeline.name},
    "Parameters": full_definition.get("Parameters", []),
    "Steps": full_definition.get("Steps", {})
}

assets_dir = "pipeline_assets"
os.makedirs(assets_dir, exist_ok=True)
output_path = os.path.join(assets_dir, "pipeline_definition.json")
with open(output_path, "w") as f:
    json.dump(service_body, f, indent=2)

print(f"Generated {output_path} successfully.")
