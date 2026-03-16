from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

policies = {
    "4.2": "Employees must go through progressive discipline before termination.",
    "3.1": "All leave must be approved by HR."
}

policy_ids = list(policies.keys())
policy_texts = list(policies.values())

policy_embeddings = model.encode(policy_texts)

question = "Can we terminate an employee immediately for repeated lateness?"

q_embedding = model.encode([question])[0]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine(q_embedding, p) for p in policy_embeddings]

best_index = int(np.argmax(scores))

print("\nUser Question:")
print(question)

print("\nMatched Policy Section:")
print(policy_ids[best_index])
print(policy_texts[best_index])

if scores[best_index] > 0.75:
    print("\n🟢 Reliability Verified")
else:
    print("\n⚠️ Orientation Drift Detected")
