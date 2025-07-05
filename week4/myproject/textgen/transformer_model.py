from transformers import AutoTokenizer, AutoModelForCausalLM

class GPT2Generator:
    def __init__(self):
        self.model_name = "koushik-25/my-finetuned-gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id  
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

generator = GPT2Generator()
