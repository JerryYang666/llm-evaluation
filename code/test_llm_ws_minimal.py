import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_name = '/mnt/pan/courses/llm24/xxh584/Meta-Llama-3.1-8B-Instruct'
# model_name = '/mnt/pan/courses/llm24/xxh584/Llama-3.2-1B'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, attn_implementation = "flash_attention_2", torch_dtype=torch.bfloat16).cuda()
model.eval()

input_text = "What is the capital of France? The answer is: "
input_ids = tokenizer.encode(input_text, return_tensors='pt')
input_ids = input_ids.cuda()

# Run the model
tokens = model.generate(input_ids, max_new_tokens=20, use_cache=True)
output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
# Open a file in write mode. If the file doesn't exist, it will be created.
with open("example_file.txt", "w") as file:
    # Write some text to the file
    file.write("Hello, this is a randomly generated text file created by a Python script.\n")
    file.write("Here's another line of text added for fun!\n")
print("File has been created and text has been written to it.")
