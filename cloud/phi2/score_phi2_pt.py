import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def init():
  torch.set_default_device("cuda")

  global model, tokenizer

  model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

def run(payload):
  
  data = json.loads(payload)

  inputs = tokenizer(data["prompt"], return_tensors="pt", return_attention_mask=False)

  outputs = model.generate(**inputs, max_length=200)
  text = tokenizer.batch_decode(outputs)[0]
  print(text)

if __name__ == '__main__':
    init()

    prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

    payload = json.dumps({"prompt": prompt})

    print(run(payload))
