import torch

class TextDataset:
    def __init__(self, path, block_size):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.data = list(text.encode("utf-8"))
        self.block_size = block_size

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))

        x, y = [], []

        for i in ix:
            chunk = self.data[i:i+self.block_size]
            target = self.data[i+1:i+self.block_size+1]

            x.append(chunk)
            y.append(target)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )