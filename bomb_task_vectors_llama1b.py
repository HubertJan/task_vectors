from transformers import AutoTokenizer, LlamaForCausalLM
from src.task_vectors import TaskVector
from huggingface_hub import login

login()

def example_inference(model):
    text = "Foreigners are "
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generated_text

pretrained = LlamaForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
finetuned = LlamaForCausalLM.from_pretrained('isaacberlin/Llama-3.2-Rightleaning')
print("Loaded models")
tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')

task_vector = TaskVector(pretrained.base_model, finetuned.base_model)
print("Created task vector")

del pretrained
finetuned_text = example_inference(finetuned)
print("")
print("")
print("Finetuned: ", finetuned_text)
del finetuned

to_be_trained = LlamaForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
task_vector = task_vector * 0
task_vector.apply_to(to_be_trained.base_model)

applied_text = example_inference(to_be_trained)

print("")
print("")
print("Finetuned: ", finetuned_text)
print("Applied: ", applied_text)