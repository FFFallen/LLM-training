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
        length = max(0, len(self.tokens) - self.block_size)
        return length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = get_tokenizer()

    # 加载数据
    print("加载数据...")

    # 加载本地 CSV 文件
    dataset_hf = load_dataset("csv", data_files="data/sample-raw-wikitext-data.csv")

    # 取出其中的文本列
    text_data = dataset_hf["train"]["text"]
    clean_text_data = [t for t in text_data if t is not None and t.strip() != ""]

    # 拼接成一个长文本（用换行符分隔）
    text = "\n".join(clean_text_data)
    print("数据加载完成，文本长度：", len(text))

    # 创建数据集和数据加载器
    dataset = TextDataset(text, tokenizer, block_size=128)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 创建模型和优化器
    model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 训练模型
    epochs = 5
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

    torch.save(model.state_dict(), "output/minigpt-5e.pt")
    print("模型训练完成！参数已保存至 output/minigpt-5e.pt")