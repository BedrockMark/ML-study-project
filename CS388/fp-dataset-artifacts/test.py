import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# 1. Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ckpt = "path/to/your/fine-tuned-electra-small" 
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)
model.eval()

# 2. Predict helper
def predict(premises, hypotheses):
    inputs = tokenizer(
        premises, hypotheses, 
        padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).cpu().numpy()

# 3. Load contrast set (e.g., from CSV/JSON)
# Expected columns: premise_orig, hypo_orig, label_orig, premise_cont, hypo_cont, label_cont, category
df = pd.read_json("lit_snli_contrast.json") 

# 4. Run inference
preds_orig = predict(df["premise_orig"].tolist(), df["hypo_orig"].tolist())
preds_cont = predict(df["premise_cont"].tolist(), df["hypo_cont"].tolist())

# 5. Compute metrics
correct_orig = (preds_orig == df["label_orig"].to_numpy())
correct_cont = (preds_cont == df["label_cont"].to_numpy())
consistency = (correct_orig & correct_cont)

print(f"Original Accuracy: {correct_orig.mean() * 100:.2f}%")
print(f"Contrast Accuracy: {correct_cont.mean() * 100:.2f}%")
print(f"Accuracy Drop:     {(correct_orig.mean() - correct_cont.mean()) * 100:.2f}%")
print(f"Consistency Score: {consistency.mean() * 100:.2f}%")

# 6. Fine-grained Breakdown by transformation category
df["orig_correct"] = correct_orig
df["cont_correct"] = correct_cont
df["consistent"] = consistency

breakdown = df.groupby("category")[["orig_correct", "cont_correct", "consistent"]].mean() * 100
print("\nBreakdown by Category:\n", breakdown)