import torch
from config import Config
from model import GPTMini

config = Config()
config.init()

model = GPTMini(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location=config.device))
model.eval()

def encode(text):
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long).unsqueeze(0)

def decode(tokens):
    return bytes(tokens.tolist()).decode("utf-8", errors="ignore")

while True:
    user = input("You: ")
    if user.lower() == "exit":
        break

    x = encode(user).to(config.device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=-1)[0]

    print("AI:", decode(pred))