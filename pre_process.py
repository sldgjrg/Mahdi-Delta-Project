#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import os
import boto3

AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")
RAW_BUCKET     = os.getenv("RAW_BUCKET")

def create_bucket(s3, name, region):
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=name)
        else:
            s3.create_bucket(
                Bucket=name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print(f"✅ Bucket {name} exists or created")
    except Exception as e:
        print(f"❌ Could not create/access bucket {name}: {e}")

def upload(s3, local_path, bucket, key):
    s3.upload_file(local_path, bucket, key)
    print(f"📤 Uploaded {local_path} → s3://{bucket}/{key}")

def main():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    create_bucket(s3, RAW_BUCKET, AWS_REGION)
    upload(s3, "Tweets.csv", RAW_BUCKET, "data/raw/Tweets.csv")

if __name__ == "__main__":
    main()
