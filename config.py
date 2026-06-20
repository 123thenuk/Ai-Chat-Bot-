import torch

class Config:
    

    
    cpu_threads = 16
    num_workers = 4
    pin_memory = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # force options
    force_cpu = False
    force_gpu = False

   
    mode = "max"

    def apply_mode(self):
        if self.mode == "fast":
            self.batch_size = 8
            self.block_size = 64
            self.n_layer = 2
            self.n_embd = 128

        elif self.mode == "balanced":
            self.batch_size = 16
            self.block_size = 128
            self.n_layer = 4
            self.n_embd = 256

        elif self.mode == "max":
            self.batch_size = 32
            self.block_size = 256
            self.n_layer = 6
            self.n_embd = 384



    vocab_size = 256
    n_head = 6



    epochs = 10
    lr = 2e-4
    grad_clip = 1.0
    weight_decay = 0.01



    data_path = "sample_data.txt"
    save_path = "model.pt"



    def init(self):
        self.apply_mode()

        if self.force_cpu:
            self.device = "cpu"
        elif self.force_gpu and torch.cuda.is_available():
            self.device = "cuda"

        print("\n===== CONFIG =====")
        print("Device:", self.device)
        print("Mode:", self.mode)
        print("Batch:", self.batch_size)
        print("Block:", self.block_size)
        print("==================\n")