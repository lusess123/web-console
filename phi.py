import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
def  phiChart(txt):
    inputs = tokenizer(txt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.batch_decode(outputs)[0]
    return text
