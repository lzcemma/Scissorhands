from datasets import load_dataset
from tqdm import tqdm
import json
import transformers


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


dataset = load_dataset("c4", "en", split="validation", streaming=True)
dataset = dataset.shuffle(buffer_size=10000, seed=42)
path = "c4_valid_min1024.jsonl"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "facebook/opt-66b", revision="main"
)
counter = 0
for idx, doc in enumerate(tqdm(dataset)):
    input = tokenizer.encode(doc["text"], return_tensors="pt")
    if input.shape[1] < 1024:
        continue
    else:
        counter += 1
        data = {
            "best_of": 1,
            "echo": True,
            "logprobs": 1,
            "max_tokens": 10,
            "model": "opt-175b",
            "n": 1,
            "prompt": doc["text"],
            "request_type": "language-model-inference",
            "stop": None,
            "temperature": 0,
            "top_p": 1,
        }
        dump_jsonl([data], path, append=True)
    if counter > 500:
        break
