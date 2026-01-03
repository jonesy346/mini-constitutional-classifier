import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

BASE = "google/gemma-2b-it" # "microsoft/phi-2" for phi2 model; "gpt2" for GPT2 model
ADAPTER = "outputs/lora-sft"

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE)
# attach LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER)

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model, tokenizer=tok, device=device)

prompt = "Instruction:\nExplain what model alignment means in simple terms.\n\nResponse:\n"
print(pipe(prompt, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.8)[0]["generated_text"])