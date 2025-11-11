from transformers import AutoTokenizer

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("tokenizer:", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer after setting pad_token:", tokenizer)
    return tokenizer