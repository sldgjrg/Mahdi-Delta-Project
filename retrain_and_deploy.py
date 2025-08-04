
import os
import tarfile
from datetime import datetime

import boto3
import botocore
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from sagemaker import Session
from sagemaker.model import ModelPackage

load_dotenv()

AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
PROCESSED_BUCKET  = os.getenv("PROCESSED_BUCKET")
ROLE_ARN          = os.getenv("SAGEMAKER_ROLE_ARN")
MODEL_GROUP_NAME  = os.getenv("MODEL_GROUP_NAME", "SentimentModelGroup")
ENDPOINT_NAME     = os.getenv("SAGEMAKER_ENDPOINT", "SentimentEndpoint")
INSTANCE_TYPE     = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.t2.medium")

s3   = boto3.client("s3", region_name=AWS_REGION)
glue = boto3.client("glue", region_name=AWS_REGION)
sm   = boto3.client("sagemaker", region_name=AWS_REGION)
sess = Session(boto_session=boto3.Session(region_name=AWS_REGION))


def run_glue_etl_job():
    try:
        print("Triggering Glue ETL job SentimentDataETL …")
        glue.start_job_run(JobName="SentimentDataETL")
    except Exception as e:
        print("Could not start Glue job:", e)



def fetch_processed_files():
    for split in ("train.csv", "test.csv"):
        remote_key = f"data/processed/{split}"
        print(f"Downloading s3://{PROCESSED_BUCKET}/{remote_key} → ./{split}")
        s3.download_file(PROCESSED_BUCKET, remote_key, split)


def local_model_exists():
    return os.path.exists("model.tar.gz")


def train_and_package():
    df_train = pd.read_csv("train.csv")
    df_test  = pd.read_csv("test.csv")
    ds_train = Dataset.from_pandas(df_train)
    ds_test  = Dataset.from_pandas(df_test)

    model_id = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)

    def tok(batch):
        return tokenizer(batch["text"], text_pair=batch["aspect"], truncation=True)

    ds_train = ds_train.map(tok, batched=True)
    ds_test  = ds_test.map(tok, batched=True)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=50,
        save_total_limit=1
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds  = np.argmax(pred.predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        return {
            "accuracy": (preds == labels).mean(),
            "precision": precision,
            "recall":    recall,
            "f1":         f1
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Eval metrics:", trainer.evaluate())

    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model", use_fast=False)

    os.makedirs("code", exist_ok=True)
    import shutil
    shutil.copy("inference.py", "code/inference.py")
    with open("code/requirements.txt", "w") as f:
        f.write("torch>=1.9.0\ntransformers>=4.21.0\n")

    with tarfile.open("model.tar.gz", "w:gz") as t:
        for fn in os.listdir("./final_model"):
            t.add(os.path.join("./final_model", fn), arcname=fn)
        t.add("code/inference.py", arcname="code/inference.py")
        t.add("code/requirements.txt", arcname="code/requirements.txt")

    print("Uploading model.tar.gz to S3")
    s3.upload_file("model.tar.gz", PROCESSED_BUCKET, "model/model.tar.gz")


def cleanup_existing_resources():
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted endpoint {ENDPOINT_NAME}")
    except sm.exceptions.ClientError:
        pass

    try:
        sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print(f"Deleted endpoint config {ENDPOINT_NAME}")
    except sm.exceptions.ClientError:
        pass


def register_and_deploy():
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_GROUP_NAME,
            ModelPackageGroupDescription="ABSA sentiment group"
        )
    except botocore.exceptions.ClientError as e:
        if "already exists" not in str(e):
            raise

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resp = sm.create_model_package(
        ModelPackageGroupName=MODEL_GROUP_NAME,
        ModelPackageDescription=f"ABSA custom inference {timestamp}",
        InferenceSpecification={
            "Containers": [{
                "Image": (
                    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                    "huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
                ),
                "ModelDataUrl": f"s3://{PROCESSED_BUCKET}/model/model.tar.gz",
                "Environment": {
                    "SAGEMAKER_PROGRAM":          "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
                }
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [INSTANCE_TYPE]
        },
        ModelApprovalStatus="Approved"
    )
    pkg_arn = resp["ModelPackageArn"]
    print("Registered model package:", pkg_arn)

    mp = ModelPackage(
        role=ROLE_ARN,
        model_package_arn=pkg_arn,
        sagemaker_session=sess
    )
    predictor = mp.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        endpoint_config_name=f"{ENDPOINT_NAME}-cfg-{timestamp}"
    )
    print(f"Deployed endpoint {ENDPOINT_NAME}")
    return predictor


if __name__ == "__main__":
    run_glue_etl_job()
    fetch_processed_files()

    if not local_model_exists():
        train_and_package()
    else:
        print("model.tar.gz exists, skipping training")

    cleanup_existing_resources()
    register_and_deploy()

