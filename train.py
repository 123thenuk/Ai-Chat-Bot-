import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from model import GPTMini
from dataset import TextDataset


checkpoint_path = "model/model.pt"


os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

torch.set_num_threads(16)
torch.set_num_interop_threads(16)



torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



config = Config()
config.init()

dataset = TextDataset(config.data_path, config.block_size)



model = GPTMini(config).to(config.device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay
)

loss_fn = nn.CrossEntropyLoss()



use_gpu = torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda') if use_gpu else None



for epoch in range(config.epochs):
    model.train()

    print(f"\n🚀 Epoch {epoch+1}/{config.epochs}")

    start_time = time.time()

    progress = tqdm(range(200), desc="Training", leave=True)

    for step in progress:

        x, y = dataset.get_batch(config.batch_size)
        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()

        
        
        if use_gpu:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        
        
        else:
            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

        
        
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "device": config.device
        })

    end_time = time.time()

    print(f"⏱ Epoch time: {end_time - start_time:.2f}s")



torch.save(model.state_dict(), config.save_path)
print("\n✅ Model saved successfully!")