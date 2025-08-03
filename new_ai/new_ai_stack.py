import json
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sm,
)
from aws_cdk.aws_s3_deployment import BucketDeployment, Source

class NewAiStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        raw_bucket  = s3.Bucket(self, "RawDataBucket")
        proc_bucket = s3.Bucket(self, "ProcessedDataBucket")

        # Upload pipeline JSON
        pipeline_deploy = BucketDeployment(
            self, "PipelineDefDeploy",
            sources=[Source.asset("pipeline_assets")],
            destination_bucket=proc_bucket,
            destination_key_prefix="definitions"
        )

        # SageMaker execution role
        role = iam.Role(
            self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )

        # Raw data bucket: read only
        raw_bucket.grant_read(role)

        # Processed bucket: read + get-bucket-location
        proc_bucket.grant_read(role)
        role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetBucketLocation"],
            resources=[proc_bucket.bucket_arn]
        ))

        # Pipeline definition in S3
        pipeline = sm.CfnPipeline(
            self, "SentimentPipeline",
            pipeline_name="SentimentAnalysisPipeline",
            role_arn=role.role_arn,
            pipeline_definition={
                "PipelineDefinitionS3Location": {
                    "Bucket": proc_bucket.bucket_name,
                    "Key":    "definitions/pipeline_definition.json"
                }
            }
        )
        pipeline.node.add_dependency(pipeline_deploy)

        # Model registry
        sm.CfnModelPackageGroup(
            self, "ModelGroup",
            model_package_group_name="SentimentModelGroup"
        )
