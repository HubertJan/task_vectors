from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
pretrained = GPT2LMHeadModel.from_pretrained('gpt2')
finetuned = GPT2LMHeadModel.from_pretrained('smgriffin/pop-lyrics-generator-v1')


def example_inference(model):
    text = "Message: "
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

text = example_inference(pretrained)
song_text = example_inference(finetuned)

print("")
print("")
print("Text: ", text)
print("Song: ", song_text)