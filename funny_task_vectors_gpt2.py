from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.task_vectors import TaskVector


pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
finetuned = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="smgriffin/pop-lyrics-generator-v1")
print("Loaded models")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

task_vector = TaskVector(pretrained.base_model, finetuned.base_model)
print("Created task vector")

strong_vector = task_vector * -0.5

to_be_trained = GPT2LMHeadModel.from_pretrained("gpt2")
strong_vector.apply_to(to_be_trained.base_model)


def example_inference(model):
    text = "Write a text: "
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

finetuned_song = example_inference(finetuned)
applied_song = example_inference(to_be_trained)

print("")
print("")
print("Finetuned: ", finetuned_song)
print("Applied: ", applied_song)