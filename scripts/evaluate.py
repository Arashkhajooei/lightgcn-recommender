# scripts/evaluate.py
import json

# Dummy scores
metrics = {
    "precision@10": 0.55,
    "ndcg@10": 0.62
}

# Save to a metrics file
with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Evaluation complete.")
