from transformers import AutoModelForCausalLM
from src.task_vectors import TaskVector


pretrained = AutoModelForCausalLM.from_pretrained("gpt2")
finetuned = AutoModelForCausalLM.from_pretrained("smgriffin/pop-lyrics-generator-v1")


task_vector = TaskVector(pretrained, finetuned)