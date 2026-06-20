# 🧠 AI Trainer (PyTorch Local Model)

This project is a local AI training system built using PyTorch.  
It allows you to train a small Transformer-based language model on your own text data directly from your PC.
### THIS ALSO INCLUDED A DATA SET TO TRAIN THE ENGLISH LANGUAGE TO THE AI BOT
---

## 👨‍💻 Author

Made by **Chubby**

---

## 🚀 What is this?

This is a mini AI training framework that lets you:

- Train your own AI model locally
- Adjust performance (CPU/GPU usage)
- Resume training from previous sessions
- Save and load trained models
- Chat with your trained AI in terminal

---

## ⚙️ Features

- Transformer-based language model
- CPU + GPU support (auto detect)
- Resume training (checkpoint system)
- Live training progress
- Model saving system
- Terminal chat system

---

## 📁 Project Structure

ai_trainer/
│
├── train.py        Main training script
├── model.py        AI model (Transformer)
├── dataset.py      Data loader
├── config.py       Settings (speed, size, GPU, etc.)
├── chat.py         Chat with trained AI
│
├── model/          Saved models folder
│   └── model.pt
│
└── sample_data.txt Training data

---

## 🧠 What each file does

train.py
Runs training loop and saves model

model.py
AI brain (Transformer network)

dataset.py
Loads and prepares text data

config.py
Controls AI settings (speed, batch size, model size)

chat.py
Lets you talk to trained AI in terminal

model/
Stores trained AI files

sample_data.txt
Training data (text your AI learns from)

---

## 🚀 How to use

Install requirements:
pip install -r requirements.txt

Train model:
python train.py

Chat with AI:
python chat.py

---

## 💡 Tips

- Use large text data for better results
- Increase epochs for better learning
- Use GPU if available for faster training

---

## ⚠️ Notes

- This is a small AI model (not ChatGPT level)
- CPU training is slower
- Results depend on your dataset quality

---

## 👨‍💻 Made by

Chubby
