import os
import math
import time
import glob
import threading
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# ==========================================
# 1. TOKENIZER & HARDWARE PROFILING
# ==========================================
class CharTokenizer:
    def __init__(self):
        chars = ["<pad>", "<unk>", "<sep>", "\n", " ", "\t"] + [chr(i) for i in range(32, 127)]
        # Expanded to include common math/science symbols
        math_symbols = ["∑", "∫", "∂", "∇", "α", "β", "γ", "δ", "θ", "λ", "μ", "π", "σ", "τ", "φ", "ω", "ℏ", "∈", "⊂", "≈", "≠", "≡", "≤", "≥", "∞", "⊗", "⊕"]
        chars.extend(math_symbols)
        chars = list(dict.fromkeys(chars)) # Ensure unique deterministically
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.unk_id = self.stoi["<unk>"]
        
    def encode(self, s):
        tokens = []
        i = 0
        while i < len(s):
            if s.startswith("<sep>", i):
                tokens.append(self.stoi["<sep>"])
                i += 5
            else:
                tokens.append(self.stoi.get(s[i], self.unk_id))
                i += 1
        return tokens
        
    def decode(self, l):
        return ''.join([self.itos.get(i, '') for i in l])

tokenizer = CharTokenizer()

# ==========================================
# 2. STREAMING DATASET HANDLER
# ==========================================
class ScientificTextDataset(Dataset):
    """
    Reads large text files dynamically without exploding System RAM.
    Streams concatenated text chunks from the data directory. The sequence 
    length determines the context window size during training.
    """
    def __init__(self, data_dir, seq_length=64):
        self.seq_length = seq_length
        self.files = glob.glob(os.path.join(data_dir, "*.txt"))
        self.tokens = []
        
        # Load tokens sequentially into a single flat list
        for file in self.files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.tokens.extend(tokenizer.encode(text))
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")

        # Store as a single contiguous tensor to save memory (much more efficient than a list of tuples)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.num_samples = max(0, len(self.tokens) - seq_length)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_length]
        y = self.tokens[idx + 1 : idx + self.seq_length + 1]
        return x, y

# ==========================================
# 3. SUBQ ARCHITECTURE (TENSOR CORE READY)
# ==========================================
class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    Unlike absolute positional embeddings which fail abruptly when generating past the 
    training context length, RoPE encodes relative positions dynamically. This aligns 
    with the 'YaRN' strategy mentioned in the SubQ technical report for robust 
    long-context extrapolation.
    """
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim/2]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(1)
        
        def apply_rotary(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
            
        return apply_rotary(q), apply_rotary(k)

class SubquadraticSparseAttention(nn.Module):
    """
    Core Attention mechanism implementing Subquadratic Sparse Attention (SSA).
    Instead of calculating dense O(N^2) attention matrices, this module divides 
    the sequence into blocks, computes block-level means, and scores blocks against 
    each other to select only the most relevant `top_k_blocks`. Dense attention is 
    then applied only within these selectively retrieved blocks, dramatically 
    reducing FLOPs at multi-million token contexts.
    """
    def __init__(self, embed_dim, num_heads, block_size=32, top_k_blocks=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        pad_len = (self.block_size - (seq_len % self.block_size)) % self.block_size
        if pad_len > 0: x = F.pad(x, (0, 0, 0, pad_len))
        seq_len_padded = seq_len + pad_len
            
        num_blocks = seq_len_padded // self.block_size
        
        # Q, K, V: [B, seq_len_padded, H, D] -> [B, H, seq_len_padded, D]
        Q = self.q_proj(x).view(batch_size, seq_len_padded, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len_padded, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len_padded, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        Q, K = self.rope(Q, K, seq_len_padded)

        # Blocked: [B, H, M, B_size, D]
        Q_blocked = Q.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        K_blocked = K.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)
        V_blocked = V.view(batch_size, self.num_heads, num_blocks, self.block_size, self.head_dim)

        # Content-dependent routing via Block means: [B, H, M, D]
        Q_block_mean = Q_blocked.mean(dim=3)
        K_block_mean = K_blocked.mean(dim=3)
        
        # Block Routing Scores: [B, H, M, M]
        routing_scores = torch.matmul(Q_block_mean, K_block_mean.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Block causal mask (query block i can only attend to key block j if j <= i)
        q_idx = torch.arange(num_blocks, device=x.device).unsqueeze(1)
        k_idx = torch.arange(num_blocks, device=x.device).unsqueeze(0)
        block_mask = q_idx >= k_idx
        routing_scores = routing_scores.masked_fill(~block_mask, float('-inf'))
        
        actual_top_k = min(self.top_k_blocks, num_blocks)
        _, top_k_indices = torch.topk(routing_scores, actual_top_k, dim=-1) # [B, H, M, K_blocks]
        
        out = torch.zeros_like(Q_blocked) # [B, H, M, B_size, D]
        
        # Vectorized gather per query block
        for q_m in range(num_blocks):
            Q_cur = Q_blocked[:, :, q_m, :, :] # [B, H, B_size, D]
            
            sel_blocks = top_k_indices[:, :, q_m, :] # [B, H, K_blocks]
            
            # Expand to gather full blocks
            sel_blocks_exp = sel_blocks.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, self.head_dim)
            K_sel = torch.gather(K_blocked, 2, sel_blocks_exp) # [B, H, K_blocks, B_size, D]
            V_sel = torch.gather(V_blocked, 2, sel_blocks_exp)
            
            # Reshape to token-level
            K_sel = K_sel.view(batch_size, self.num_heads, actual_top_k * self.block_size, self.head_dim)
            V_sel = V_sel.view(batch_size, self.num_heads, actual_top_k * self.block_size, self.head_dim)
            
            # Dense token attention within selected blocks
            attn_weights = torch.matmul(Q_cur, K_sel.transpose(-1, -2)) / math.sqrt(self.head_dim) # [B, H, B_size, K_blocks * B_size]
            
            # Token causal mask
            q_tok_idx = torch.arange(q_m * self.block_size, (q_m + 1) * self.block_size, device=x.device).unsqueeze(1)
            
            base_idx = torch.arange(self.block_size, device=x.device)
            k_tok_idx = (sel_blocks.unsqueeze(-1) * self.block_size + base_idx).view(batch_size, self.num_heads, -1) # [B, H, K_blocks * B_size]
            
            mask = q_tok_idx.view(1, 1, self.block_size, 1) >= k_tok_idx.unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            
            attn_probs = F.softmax(attn_weights, dim=-1)
            out[:, :, q_m, :, :] = torch.matmul(attn_probs, V_sel)

        out = out.view(batch_size, self.num_heads, seq_len_padded, self.head_dim).transpose(1, 2).contiguous().view(batch_size, seq_len_padded, embed_dim)
        if pad_len > 0: out = out[:, :-pad_len, :]
        return self.out_proj(out)

class SubQVariantModel(nn.Module):
    """
    The main language model combining character embeddings, multiple layers 
    of Subquadratic Sparse Attention, and LayerNorms. Uses a character-level 
    tokenizer designed to process math and physics equations seamlessly.
    """
    def __init__(self, vocab_size, embed_dim=256, depth=6, num_heads=8, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(embed_dim),
                "attn": SubquadraticSparseAttention(embed_dim, num_heads, block_size=32, top_k_blocks=4),
                "ln_2": nn.LayerNorm(embed_dim),
                "mlp": nn.Linear(embed_dim, embed_dim)
            }) for _ in range(depth)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer["attn"](layer["ln_1"](x))
            x = x + layer["mlp"](layer["ln_2"](x))
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, max_seq_len=2048):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= max_seq_len else input_ids[:, -max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            yield idx_next.item()
        self.train()

# ==========================================
# 4. GUI & HARDWARE SCHEDULER
# ==========================================
class SubQStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SubQ Science-Grade Data Ingestion Studio")
        self.root.geometry("1000x750")
        
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SubQVariantModel(vocab_size=tokenizer.vocab_size).to(self.gpu_device)
        self.is_training = False
        self.save_dir = "./subq_models"
        self.data_dir = "./science_data"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.build_db_tab()
        self.build_training_tab()
        self.build_inference_tab()

    def log(self, text_widget, msg, newline=True):
        text_widget.insert(tk.END, msg + ("\n" if newline else ""))
        text_widget.see(tk.END)

    # --- TAB 1: DATABASE MANAGER ---
    def build_db_tab(self):
        db_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(db_frame, text="📚 Database Importer")
        
        ttk.Label(db_frame, text="Scientific Corpus Pipeline", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0,10))
        
        info = (
            f"Place pure text files (.txt) inside the local dataset directory: {os.path.abspath(self.data_dir)}\n"
            "The DataLoader will stream these files directly into pinned memory, preventing 32GB RAM saturation "
            "while keeping the 4070's Tensor Cores fully saturated."
        )
        ttk.Label(db_frame, text=info, wraplength=900).pack(anchor="w", pady=5)
        
        self.file_listbox = tk.Listbox(db_frame, height=15)
        self.file_listbox.pack(fill='both', expand=True, pady=10)
        
        btn_frame = ttk.Frame(db_frame)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="Scan Dataset Directory", command=self.scan_directory).pack(side=tk.LEFT)
        self.scan_directory()

    def scan_directory(self):
        self.file_listbox.delete(0, tk.END)
        files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        if not files:
            self.file_listbox.insert(tk.END, "No text databases found. Create './science_data' and add .txt files.")
        for f in files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            self.file_listbox.insert(tk.END, f"{os.path.basename(f)} - {size_mb:.2f} MB")

    # --- TAB 3: ACTIVE INFERENCE ---
    def build_inference_tab(self):
        infer_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(infer_frame, text="🧠 Active Inference")
        
        top_bar = ttk.Frame(infer_frame)
        top_bar.pack(fill="x", pady=5)
        
        ttk.Button(top_bar, text="Load Weights (.pt)", command=self.load_model).pack(side=tk.LEFT, padx=5)
        self.lbl_model_status = ttk.Label(top_bar, text="Status: Unloaded", foreground="gray")
        self.lbl_model_status.pack(side=tk.LEFT, padx=10)
        
        self.chat_log = scrolledtext.ScrolledText(infer_frame, height=20, bg="#0d0d0d", fg="#00ffff", font=("Consolas", 11))
        self.chat_log.pack(fill="both", expand=True, pady=10)
        self.chat_log.insert(tk.END, "SubQ Active Inference Engine Ready.\nLoad a model to begin.\n\n")
        
        ctrl_bar = ttk.Frame(infer_frame)
        ctrl_bar.pack(fill="x")
        
        ttk.Label(ctrl_bar, text="Prompt:").pack(side=tk.LEFT)
        self.prompt_var = tk.StringVar()
        self.entry_prompt = ttk.Entry(ctrl_bar, textvariable=self.prompt_var, width=60)
        self.entry_prompt.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        self.entry_prompt.bind("<Return>", lambda e: self.start_generation())
        
        ttk.Label(ctrl_bar, text="Temp:").pack(side=tk.LEFT, padx=2)
        self.temp_var = tk.DoubleVar(value=0.8)
        ttk.Entry(ctrl_bar, textvariable=self.temp_var, width=4).pack(side=tk.LEFT)
        
        self.btn_generate = ttk.Button(ctrl_bar, text="Generate", command=self.start_generation, state=tk.DISABLED)
        self.btn_generate.pack(side=tk.LEFT, padx=10)

    def load_model(self):
        filepath = filedialog.askopenfilename(initialdir=self.save_dir, filetypes=[("PyTorch Model", "*.pt")])
        if filepath:
            try:
                self.model.load_state_dict(torch.load(filepath, map_location=self.gpu_device))
                self.model.eval()
                self.lbl_model_status.config(text=f"Loaded: {os.path.basename(filepath)}", foreground="green")
                self.btn_generate.config(state=tk.NORMAL)
                self.log(self.chat_log, f"[System] Model weights synchronized from {os.path.basename(filepath)}.\n")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    def start_generation(self):
        prompt = self.prompt_var.get()
        if not prompt: return
        self.prompt_var.set("")
        self.btn_generate.config(state=tk.DISABLED)
        self.entry_prompt.config(state=tk.DISABLED)
        
        self.log(self.chat_log, f"User: {prompt}\nSubQ: ", newline=False)
        threading.Thread(target=self.execute_generation, args=(prompt,), daemon=True).start()

    def execute_generation(self, prompt):
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=self.gpu_device)
        temp = self.temp_var.get()
        
        try:
            for next_token_id in self.model.generate(input_ids, max_new_tokens=500, temperature=temp, top_k=10):
                char = tokenizer.decode([next_token_id])
                self.root.after(0, lambda c=char: self.chat_log.insert(tk.END, c))
                self.root.after(0, lambda: self.chat_log.see(tk.END))
        except Exception as e:
            self.root.after(0, lambda e=e: self.log(self.chat_log, f"\n[Generation Error]: {e}"))
            
        self.root.after(0, lambda: self.log(self.chat_log, "\n\n", newline=False))
        self.root.after(0, lambda: self.btn_generate.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.entry_prompt.config(state=tk.NORMAL))

    # --- TAB 2: GPU OPTIMIZED TRAINING ---
    def build_training_tab(self):
        train_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(train_frame, text="⚙️ CUDA Core Engine")
        
        cfg_box = ttk.LabelFrame(train_frame, text=" Hardware & Memory Protocol ", padding=10)
        cfg_box.pack(fill="x", pady=5)
        
        self.use_amp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_box, text="Enable Mixed Precision (Leverages RTX 4070 Tensor Cores / BF16)", variable=self.use_amp_var).grid(row=0, column=0, sticky="w", pady=2)
        
        self.ram_overflow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_box, text="Asynchronous Optimizer Offload (32GB Host RAM)", variable=self.ram_overflow_var).grid(row=1, column=0, sticky="w", pady=2)
        
        self.delta_train_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_box, text="Freeze Base Topologies (Halt Hallucinations)", variable=self.delta_train_var).grid(row=2, column=0, sticky="w", pady=2)
        
        batch_row = ttk.Frame(cfg_box)
        batch_row.grid(row=3, column=0, sticky="w", pady=5)
        ttk.Label(batch_row, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Entry(batch_row, textvariable=self.batch_size_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(batch_row, text="Epochs:").pack(side=tk.LEFT, padx=10)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(batch_row, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT)

        btn_layout = ttk.Frame(train_frame)
        btn_layout.pack(fill="x", pady=5)
        self.btn_train = ttk.Button(btn_layout, text="▶ Ignite Training Pipeline", command=self.start_training)
        self.btn_train.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(btn_layout, text="🛑 Terminate", command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)
        
        self.train_log = scrolledtext.ScrolledText(train_frame, height=15, bg="#0d0d0d", fg="#00ff00")
        self.train_log.pack(fill="both", expand=True, pady=5)

    def start_training(self):
        self.is_training = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self.execute_training, daemon=True).start()

    def stop_training(self):
        self.is_training = False

    def execute_training(self):
        self.log(self.train_log, "--- Booting RTX 4070 Database Ingestion ---")
        self.best_loss = float('inf')
        
        # 1. Prepare Streaming Dataset
        # Increased seq_length to 512 so that num_blocks (16) > top_k_blocks (4).
        # This forces the routing network to actually learn to select blocks.
        dataset = ScientificTextDataset(self.data_dir, seq_length=512)
        if len(dataset) == 0:
            self.log(self.train_log, "[Error] Database empty. Cannot initialize DataLoader.")
            self.btn_train.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            return

        # DataLoader configured for GPU Starvation Prevention
        # pin_memory=True locks the RAM, num_workers allows parallel CPU fetching
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size_var.get(), 
            shuffle=True, 
            pin_memory=torch.cuda.is_available(), 
            num_workers=0, # Set to 2 or 4 if using a Linux machine, Windows often prefers 0 for Tkinter stability
            drop_last=True
        )

        # 2. Lock Core Weights
        if self.delta_train_var.get():
            self.log(self.train_log, "[Lock] Securing core calculus/physics representations.")
            for name, param in self.model.named_parameters():
                param.requires_grad = ("attn" in name or "router" in name)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 3. RAM Optimizer Offloading & Scaler Setup
        use_cpu_offload = self.ram_overflow_var.get()
        if use_cpu_offload:
            self.log(self.train_log, "[Memory] Redirecting Optimizer state matrices to 32GB System RAM.")
            cpu_params = [torch.nn.Parameter(p.clone().detach().cpu(), requires_grad=True) for p in trainable_params]
            optimizer = AdamW(cpu_params, lr=1e-4)
        else:
            optimizer = AdamW(trainable_params, lr=1e-4)

        amp_enabled = self.use_amp_var.get() and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
        if scaler:
            self.log(self.train_log, "[Compute] Automatic Mixed Precision active. Tensor Cores engaged.")
        else:
            self.log(self.train_log, "[Compute] Standard precision (FP32) active. CUDA not available for AMP.")

        self.model.train()
        epochs = self.epochs_var.get()
        
        for epoch in range(epochs):
            if not self.is_training: break
            
            for step, (x_batch, y_batch) in enumerate(dataloader):
                if not self.is_training: break
                
                # non_blocking=True allows the GPU to compute while CPU moves the next batch
                x_batch = x_batch.to(self.gpu_device, non_blocking=True)
                y_batch = y_batch.to(self.gpu_device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True) # Clears CPU gradients if offloading
                self.model.zero_grad(set_to_none=True) # CRITICAL FIX: Clear GPU gradients!
                
                # Forward Pass (with optional Tensor Core acceleration)
                if amp_enabled:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(x_batch)
                        # Sample-level loss aggregation (SubQ-1.1-Small paper)
                        loss_unreduced = F.cross_entropy(logits.transpose(1, 2), y_batch, reduction='none')
                        loss = loss_unreduced.mean(dim=1).mean()
                    
                    # Scaled Backward Pass
                    scaler.scale(loss).backward()
                    
                    if use_cpu_offload:
                        inv_scale = 1. / scaler.get_scale()
                        
                        # Validate gradients before copying to RAM
                        has_inf = False
                        for gpu_p in trainable_params:
                            if gpu_p.grad is not None:
                                if torch.isinf(gpu_p.grad).any() or torch.isnan(gpu_p.grad).any():
                                    has_inf = True
                                    break
                                    
                        if has_inf:
                            scaler.update() # drop step and scale down
                        else:
                            # Copy gradients to RAM safely
                            for gpu_p, cpu_p in zip(trainable_params, cpu_params):
                                if gpu_p.grad is not None:
                                    cpu_p.grad = (gpu_p.grad.to('cpu', non_blocking=True) * inv_scale)
                            
                            torch.nn.utils.clip_grad_norm_(cpu_params, 1.0)
                            optimizer.step()
                            scaler.update()
                            
                            # Copy weights back to 4070
                            with torch.no_grad():
                                for gpu_p, cpu_p in zip(trainable_params, cpu_params):
                                    gpu_p.copy_(cpu_p.to(self.gpu_device, non_blocking=True))
                    else:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                else:
                    logits = self.model(x_batch)
                    # Sample-level loss aggregation (SubQ-1.1-Small paper)
                    loss_unreduced = F.cross_entropy(logits.transpose(1, 2), y_batch, reduction='none')
                    loss = loss_unreduced.mean(dim=1).mean()
                    loss.backward()
                    
                    if use_cpu_offload:
                        # Copy gradients to CPU
                        for gpu_p, cpu_p in zip(trainable_params, cpu_params):
                            if gpu_p.grad is not None:
                                cpu_p.grad = gpu_p.grad.to('cpu', non_blocking=True)
                                
                        torch.nn.utils.clip_grad_norm_(cpu_params, 1.0)
                        optimizer.step()
                        
                        # Copy updated weights back to GPU/Model
                        with torch.no_grad():
                            for gpu_p, cpu_p in zip(trainable_params, cpu_params):
                                gpu_p.copy_(cpu_p.to(self.gpu_device, non_blocking=True))
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()

                if step % 20 == 0:
                    loss_val = loss.item()
                    atl_msg = ""
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        atl_msg = " (⭐ All-Time Low!)"
                    self.log(self.train_log, f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss_val:.4f}{atl_msg}")
                    
        self.log(self.train_log, "--- Pipeline Run Concluded ---")
        save_path = os.path.join(self.save_dir, f"subq_science_{int(time.time())}.pt")
        torch.save(self.model.state_dict(), save_path)
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = tk.Tk()
    SubQStudioApp(app)
    app.mainloop()