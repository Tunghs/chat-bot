# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-7b")
model = AutoModelForCausalLM.from_pretrained("beomi/gemma-ko-7b")
model.save('./gemma-ko-7b')