
"""
Script to clean up all SageMaker resources (endpoints, endpoint configs, models)
"""
import boto3
import botocore
from dotenv import load_dotenv
import os

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
sm = boto3.client("sagemaker", region_name=AWS_REGION)

def delete_endpoints():
    try:
        response = sm.list_endpoints()
        endpoints = response.get('Endpoints', [])
        
        if not endpoints:
            print("No endpoints found")
            return
            
        for endpoint in endpoints:
            endpoint_name = endpoint['EndpointName']
            try:
                print(f"Deleting endpoint: {endpoint_name}")
                sm.delete_endpoint(EndpointName=endpoint_name)
                print(f"Deleted endpoint: {endpoint_name}")
            except Exception as e:
                print(f"Failed to delete endpoint {endpoint_name}: {e}")
                
    except Exception as e:
        print(f"Error listing endpoints: {e}")

def delete_endpoint_configs():
    try:
        response = sm.list_endpoint_configs()
        configs = response.get('EndpointConfigs', [])
        
        if not configs:
            print("No endpoint configs found")
            return
            
        for config in configs:
            config_name = config['EndpointConfigName']
            try:
                print(f"Deleting endpoint config: {config_name}")
                sm.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"Deleted endpoint config: {config_name}")
            except Exception as e:
                print(f"Failed to delete endpoint config {config_name}: {e}")
                
    except Exception as e:
        print(f"Error listing endpoint configs: {e}")

def delete_models():
    try:
        response = sm.list_models()
        models = response.get('Models', [])
        
        if not models:
            print("No models found")
            return
            
        for model in models:
            model_name = model['ModelName']
            try:
                print(f"Deleting model: {model_name}")
                sm.delete_model(ModelName=model_name)
                print(f"Deleted model: {model_name}")
            except Exception as e:
                print(f"Failed to delete model {model_name}: {e}")
                
    except Exception as e:
        print(f"Error listing models: {e}")

def wait_for_endpoint_deletion():
    import time
    print("Waiting for endpoints to be fully deleted...")
    
    max_wait = 300
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            response = sm.list_endpoints()
            endpoints = response.get('Endpoints', [])
            
            if not endpoints:
                print("All endpoints deleted successfully")
                return True
                
            print(f"Still waiting... {len(endpoints)} endpoints remaining")
            time.sleep(10)
            wait_time += 10
            
        except Exception as e:
            print(f"Error checking endpoints: {e}")
            break
    
    print("Timeout waiting for endpoint deletion")
    return False

def main():
    print("Starting SageMaker cleanup...")
    print("=" * 50)
    delete_endpoints()
    
    if wait_for_endpoint_deletion():
        delete_endpoint_configs()
        delete_models()
    else:
        print("Skipping config and model cleanup due to endpoint deletion timeout")
    
    print("=" * 50)
    print("Cleanup completed!")

if __name__ == "__main__":
    main()