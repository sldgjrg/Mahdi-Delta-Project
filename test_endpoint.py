
"""
Simple script to test the SageMaker endpoint
"""
import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "SentimentEndpoint")

def test_endpoint():
    client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    
    # Test cases
    test_cases = [
        {"text": "The flight was terrible", "aspect": "service"},
        {"text": "Great customer support", "aspect": "support"},
        {"text": "Food was amazing", "aspect": "food"},
        {"text": "Horrible experience overall", "aspect": None}
    ]
    
    print(f"Testing endpoint: {ENDPOINT_NAME}")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        aspect = test_case["aspect"]
        
        # Format input
        if aspect:
            input_text = f"{text} [SEP] {aspect}"
        else:
            input_text = text
        
        payload = json.dumps({"inputs": input_text})
        
        print(f"Test {i}: '{text}' (aspect: {aspect or 'None'})")
        
        try:
            response = client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                Body=payload,
                ContentType="application/json"
            )
            
            raw_response = response["Body"].read().decode()
            print(f"Raw response: '{raw_response}'")
            print(f"Response type: {type(raw_response)}")
            
            try:
                result = json.loads(raw_response)
            except json.JSONDecodeError as e:
                print(f" JSON decode error: {e}")
                print(f"Raw response was: {raw_response}")
                continue
            
            print(f"Parsed result: {result}")
            print(f"Result type: {type(result)}")
            
            # Handle different response formats
            if isinstance(result, list):
                # Default Hugging Face format: [{"label": "NEGATIVE", "score": 0.99}]
                if result:
                    best_result = max(result, key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0)
                    if isinstance(best_result, dict):
                        print(f" Result: {best_result.get('label', 'N/A')} "
                              f"(confidence: {best_result.get('score', 0):.3f})")
                        
                        # Show all scores
                        score_str = ", ".join([f"{r.get('label', 'N/A')}:{r.get('score', 0):.3f}" 
                                             for r in result if isinstance(r, dict)])
                        print(f"   All scores: {score_str}")
                    else:
                        print(f" Result: {best_result}")
                else:
                    print(" Empty result")
            
            elif isinstance(result, dict):
                # Custom format
                print(f" Result: {result.get('predicted_label', 'N/A')} "
                      f"(confidence: {result.get('confidence', 0):.3f})")
                
                if 'all_scores' in result:
                    scores = result['all_scores']
                    score_str = ", ".join([f"{s['label']}:{s['score']:.3f}" for s in scores])
                    print(f"   Scores: {score_str}")
            
            elif isinstance(result, str):
                print(f" String result: {result}")
            
            else:
                print(f" Unexpected result format: {type(result)} - {result}")
            
        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 30)

def check_endpoint_status():
    """Check if endpoint is ready"""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    
    try:
        response = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response['EndpointStatus']
        print(f"Endpoint Status: {status}")
        
        if status == 'InService':
            print("Endpoint is ready for inference")
            return True
        elif status in ['Creating', 'Updating']:
            print("Endpoint is still being created/updated")
            return False
        else:
            print(f"Endpoint status: {status}")
            return False
            
    except Exception as e:
        print(f"Error checking endpoint: {e}")
        return False

if __name__ == "__main__":
    print("Checking endpoint status...")
    
    if check_endpoint_status():
        print("Running tests...")
        test_endpoint()
    else:
        print("Endpoint not ready. Please wait and try again.")