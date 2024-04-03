import transformers

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

for model_name in ["t5-base", "sentence-transformers/sentence-t5-base"]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    model = transformers.AutoModelForPreTraining.from_pretrained(model_name)
    model.save_pretrained(f"assets/transformers/{model_name}")
