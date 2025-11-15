from transformers import AutoTokenizer

def get_tokenizer():
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer