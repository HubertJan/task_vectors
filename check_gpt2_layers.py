from transformers import AutoModelForCausalLM

def print_layer_sizes(model):
    for layer_name in model.state_dict():
        layer = model.state_dict()[layer_name]
        print(f"Layer: {layer_name}, {layer.size()}")
        
def print_layer_sizes_and_compare(model, other_model):
    for layer_name in model.state_dict():
        layer = model.state_dict()[layer_name]
        other_layer = other_model.state_dict()[layer_name]
        print(f"Layer: {layer_name}, {layer.size()} with {other_layer.size()}")


pretrained = AutoModelForCausalLM.from_pretrained("gpt2")
finetuned = AutoModelForCausalLM.from_pretrained("smgriffin/pop-lyrics-generator-v1")

print_layer_sizes_and_compare(pretrained, finetuned)