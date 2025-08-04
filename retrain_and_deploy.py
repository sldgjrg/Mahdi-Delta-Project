
import os
import tarfile
import boto3
import botocore
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
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
RAW_BUCKET        = os.getenv("RAW_BUCKET")
PROCESSED_BUCKET  = os.getenv("PROCESSED_BUCKET")
ROLE_ARN          = os.getenv("SAGEMAKER_ROLE_ARN")
MODEL_GROUP_NAME  = os.getenv("MODEL_GROUP_NAME", "SentimentModelGroup")
ENDPOINT_NAME     = os.getenv("SAGEMAKER_ENDPOINT", "SentimentEndpoint")
INSTANCE_TYPE     = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.t2.medium")

s3 = boto3.client("s3", region_name=AWS_REGION)
sm = boto3.client("sagemaker", region_name=AWS_REGION)
sess = Session(boto_session=boto3.Session(region_name=AWS_REGION))


def local_model_exists():
    return os.path.exists("model.tar.gz")

def download_and_split():
    obj = s3.get_object(Bucket=RAW_BUCKET, Key="data/raw/Tweets.csv")
    df  = pd.read_csv(obj["Body"])
    df = df.dropna(subset=["text","airline_sentiment","airline"])
    df = df.rename(columns={"airline":"aspect","airline_sentiment":"label"})
    df["label"] = df["label"].map({"negative":0,"neutral":1,"positive":2})

    train, test = train_test_split(
        df, test_size=0.05, random_state=42, stratify=df["label"]
    )
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv",  index=False)
    print(f" Split into {len(train)} train / {len(test)} test")
    s3.upload_file("train.csv", PROCESSED_BUCKET, "data/processed/train.csv")
    s3.upload_file("test.csv",  PROCESSED_BUCKET, "data/processed/test.csv")
    print("📤 Uploaded train/test CSVs")



def train_and_package():
    df_train = pd.read_csv("train.csv")
    df_test  = pd.read_csv("test.csv")
    ds_train = Dataset.from_pandas(df_train)
    ds_test  = Dataset.from_pandas(df_test)

    model_id  = "yangheng/deberta-v3-base-absa-v1.1"
    
    # CRITICAL FIX: Use slow tokenizer to avoid serialization issues
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3
    )

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
    print("Eval:", trainer.evaluate())

    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model", use_fast=False)
    
    os.makedirs("code", exist_ok=True)
    import shutil
    shutil.copy("inference.py", "code/inference.py")
    requirements = """torch>=1.9.0
transformers>=4.21.0
"""
    
    with open("code/requirements.txt", "w") as f:
        f.write(requirements)
    
    with tarfile.open("model.tar.gz", "w:gz") as t:
        for fn in os.listdir("./final_model"):
            t.add(os.path.join("./final_model", fn), arcname=fn)
        
        t.add("code/inference.py", arcname="code/inference.py")
        t.add("code/requirements.txt", arcname="code/requirements.txt")
    
    print(" Created model.tar.gz with custom inference code")

    s3.upload_file("model.tar.gz", PROCESSED_BUCKET, "model/model.tar.gz")
    print("Uploaded model.tar.gz to S3")

def cleanup_existing_resources():
    """Clean up existing endpoint and endpoint config if they exist"""
    
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Deleting existing endpoint: {ENDPOINT_NAME}")
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        
        print(" Waiting for endpoint deletion...")
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
                print(f" Still deleting... ({wait_time}s)")
                time.sleep(10)
                wait_time += 10
            except sm.exceptions.ClientError as e:
                if "does not exist" in str(e):
                    print(" Endpoint deleted successfully")
                    break
                else:
                    raise
        else:
            print("  Timeout waiting for endpoint deletion")
            
    except sm.exceptions.ClientError as e:
        if "does not exist" in str(e):
            print(f"  Endpoint {ENDPOINT_NAME} doesn't exist, skipping deletion")
        else:
            print(f" Error checking endpoint: {e}")
    
    try:
        sm.describe_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print(f"  Deleting existing endpoint config: {ENDPOINT_NAME}")
        sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print(" Endpoint config deleted")
    except sm.exceptions.ClientError as e:
        if "does not exist" in str(e):
            print(f"  Endpoint config {ENDPOINT_NAME} doesn't exist, skipping deletion")
        else:
            print(f" Error checking endpoint config: {e}")

def register_and_deploy():
    s3_model = f"s3://{PROCESSED_BUCKET}/model/model.tar.gz"

    cleanup_existing_resources()

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
        ModelPackageDescription=f"ABSA with custom inference {timestamp}",
        InferenceSpecification={
            "Containers":[
                {
                    "Image": (
                        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                        "huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04"
                    ),
                    "ModelDataUrl": s3_model,
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
                    }
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [INSTANCE_TYPE]
        },
        ModelApprovalStatus="Approved"
    )
    pkg_arn = resp["ModelPackageArn"]
    print(" Registered model package:", pkg_arn)

    endpoint_config_name = f"{ENDPOINT_NAME}-{timestamp}"
    
    mp = ModelPackage(
        role=ROLE_ARN,
        model_package_arn=pkg_arn,
        sagemaker_session=sess
    )

    predictor = mp.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        endpoint_config_name=endpoint_config_name
    )
    print(" Created new endpoint:", ENDPOINT_NAME)
    return predictor

if __name__ == "__main__":
    if not local_model_exists():
        print(" No local model.tar.gz found. Training now…")
        download_and_split()
        train_and_package()
    else:
        print(" Found local model.tar.gz. Skipping training.")
    
    predictor = register_and_deploy()
    print(f" Deployment complete! Endpoint: {ENDPOINT_NAME}")