#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import os
import json
import torch
import argparse
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification

AWS_REGION    = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR     = os.getenv("ABSA_MODEL_DIR", "./final_model")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT")

def load_local():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.eval()
    return tok, mdl

def predict_local(tok, mdl, text, aspect):
    input_text = f"{text} [SEP] {aspect}" if aspect else text
    inputs = tok(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = mdl(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        id2lab = mdl.config.id2label
        idx = int(probs.argmax())
    return id2lab[idx], float(probs[idx]), {id2lab[i]: float(probs[i]) for i in range(len(probs))}

def invoke_sagemaker(text, aspect):
    client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    input_text = f"{text} [SEP] {aspect}" if aspect else text
    payload = json.dumps({"inputs": input_text})

    try:
        resp = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            Body=payload,
            ContentType="application/json",
            Accept="application/json"
        )
        raw = resp["Body"].read().decode()
        # 1) initial parse
        result = json.loads(raw)

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        if isinstance(result, list) and len(result) == 2 and isinstance(result[0], str):
            try:
                result = json.loads(result[0])
            except json.JSONDecodeError:
                result = result[0]
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            parsed = []
            for s in result:
                try:
                    parsed.append(json.loads(s))
                except json.JSONDecodeError:
                    parsed.append(s)
            result = parsed

        if isinstance(result, dict):
            return result

        if isinstance(result, list) and result and isinstance(result[0], dict) and "score" in result[0]:
            best = max(result, key=lambda x: x.get("score", 0))
            return {
                "predicted_label": best.get("label", "UNKNOWN"),
                "confidence": best.get("score", 0),
                "all_scores": [
                    {"label": r.get("label", "UNKNOWN"), "score": r.get("score", 0)}
                    for r in result
                ]
            }

        return {"error": "Unexpected format", "raw_response": result}

    except Exception as e:
        print(f"SageMaker inference error: {e}")
        return None

def interactive(tok=None, mdl=None, use_sm=False):
    print("Type 'quit' or 'exit' to stop.")
    while True:
        text = input("Text: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        aspect = input("Aspect (optional): ").strip()
        if use_sm:
            out = invoke_sagemaker(text, aspect)
            if out:
                print("→ SageMaker result:")
                print(f"  Label:      {out.get('predicted_label', 'N/A')}")
                print(f"  Confidence: {out.get('confidence', 0):.4f}")
                if "all_scores" in out:
                    print("  All scores:")
                    for s in out["all_scores"]:
                        print(f"    {s['label']}: {s['score']:.3f}")
            else:
                print("→ SageMaker inference failed")
        else:
            lbl, sc, scores = predict_local(tok, mdl, text, aspect)
            print(f"→ {lbl} ({sc:.4f})")
            print("  Scores:", ", ".join(f"{l}:{s:.3f}" for l, s in scores.items()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use-sagemaker", action="store_true",
                   help="Invoke the deployed SageMaker endpoint")
    p.add_argument("--text", help="Sentence to analyze")
    p.add_argument("--aspect", help="Aspect term (optional)")
    args = p.parse_args()

    if args.use_sagemaker:
        if not ENDPOINT_NAME:
            p.error("Set SAGEMAKER_ENDPOINT in .env first")
        if args.text:
            res = invoke_sagemaker(args.text, args.aspect)
            if res:
                print(json.dumps(res, indent=2))
        else:
            interactive(use_sm=True)
    else:
        tok, mdl = load_local()
        if args.text:
            lbl, sc, _ = predict_local(tok, mdl, args.text, args.aspect)
            print(f"{lbl} ({sc:.4f})")
        else:
            interactive(tok, mdl, use_sm=False)

if __name__ == "__main__":
    main()
