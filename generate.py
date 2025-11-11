import torch
from model import MiniGPT
from tokenizer import get_tokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = get_tokenizer()
model = MiniGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("output/minigpt.pt", map_location=device))
model.to(device)
model.eval()

prompt = "你好，请介绍一下你自己。"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=50)

generated = tokenizer.decode(output_ids[0].tolist())
print("\n生成结果：\n", generated)