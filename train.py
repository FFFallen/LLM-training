import torch
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizer import get_tokenizer
from tqdm import tqdm
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        # 调试信息
        length = max(0, len(self.tokens) - self.block_size)
        print("Dataset length:", length)
        return length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = get_tokenizer()

    # 加载数据
    print("加载数据...\n")

    dataset_hf = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    text = "\n".join(dataset_hf["train"]["text"])
    print("数据加载完成，文本长度：", len(text))
    dataset = TextDataset(text, tokenizer, block_size=128)
    print("dataset:",dataset)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    epochs = 3
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "output/minigpt.pt")
    print("模型训练完成！参数已保存至 output/minigpt.pt")