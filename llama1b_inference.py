from transformers import AutoTokenizer, LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
model = LlamaForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct')

text = "Hello, I'm a language model,"

input_ids = tokenizer.encode(text, return_tensors='pt')

output_ids = model.generate(input_ids, max_length=30, num_return_sequences=1, do_sample=True, top_k=50)


generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text: ", generated_text)
