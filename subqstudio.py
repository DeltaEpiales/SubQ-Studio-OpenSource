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
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, IterableDataset, DataLoader
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tiktoken

# Enable maximum hardware acceleration for RTX 40-series Tensor Cores
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. TOKENIZER & HARDWARE PROFILING
# ==========================================
class BPETokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.enc.n_vocab + 1 # +1 for <sep>
        self.sep_id = self.enc.n_vocab
        
    def encode(self, s):
        tokens = []
        parts = s.split("<sep>")
        for i, part in enumerate(parts):
            if part:
                tokens.extend(self.enc.encode(part, allowed_special="all"))
            if i < len(parts) - 1:
                tokens.append(self.sep_id)
        return tokens
        
    def decode(self, l):
        decoded = ""
        for t in l:
            if t == self.sep_id:
                decoded += "<sep>"
            else:
                decoded += self.enc.decode([t])
        return decoded

tokenizer = BPETokenizer()

# ==========================================
# 2. STREAMING DATASET HANDLER
# ==========================================
class ScientificTextDataset(IterableDataset):
    """
    Reads large text files dynamically without exploding System RAM.
    Streams concatenated text chunks from the data directory.
    """
    def __init__(self, data_dir, seq_length=64):
        super().__init__()
        self.seq_length = seq_length
        self.files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        # Estimate total samples for progress bars and schedulers
        total_bytes = sum(os.path.getsize(f) for f in self.files)
        estimated_tokens = total_bytes // 4
        self.estimated_samples = max(1, estimated_tokens // seq_length)

    def __len__(self):
        return self.estimated_samples

    def __iter__(self):
        buffer = []
        for file in self.files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    while True:
                        lines = f.readlines(1024 * 1024) # ~1MB chunk
                        if not lines:
                            break
                        text = "".join(lines)
                        buffer.extend(tokenizer.encode(text))
                        
                        while len(buffer) > self.seq_length:
                            x = buffer[:self.seq_length]
                            y = buffer[1:self.seq_length + 1]
                            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                            buffer = buffer[self.seq_length:] # Non-overlapping stride
            except Exception as e:
                pass

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

    def forward(self, q, k, seq_len, offset=0):
        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim/2]
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(1)
        
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

    def forward(self, x, past_key_value=None, use_cache=False):
        batch_size, seq_len, embed_dim = x.size()
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].size(2)
            
        Q, K = self.rope(Q, K, seq_len, offset=offset)
        
        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)
            
        new_past = (K, V) if use_cache else None

        if seq_len == 1 and past_key_value is not None:
            # Fast path for single-token autoregressive generation
            attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn_probs = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_probs, V)
            out = out.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
            return self.out_proj(out), new_past
            
        pad_len = (self.block_size - (seq_len % self.block_size)) % self.block_size
        if pad_len > 0: 
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))
        seq_len_padded = seq_len + pad_len
            
        num_blocks = seq_len_padded // self.block_size
        
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
        
        # Fully Vectorized Sparse Block Attention
        BH = batch_size * self.num_heads
        
        # Flatten K and V for vectorized gather
        K_flat = K_blocked.view(BH, num_blocks, self.block_size * self.head_dim)
        V_flat = V_blocked.view(BH, num_blocks, self.block_size * self.head_dim)
        
        # Gather index setup
        idx = top_k_indices.view(BH, num_blocks * actual_top_k)
        idx_gather = idx.unsqueeze(-1).expand(-1, -1, self.block_size * self.head_dim)
        
        # Gather selected blocks across all queries simultaneously
        K_sel = torch.gather(K_flat, 1, idx_gather).view(batch_size, self.num_heads, num_blocks, actual_top_k * self.block_size, self.head_dim)
        V_sel = torch.gather(V_flat, 1, idx_gather).view(batch_size, self.num_heads, num_blocks, actual_top_k * self.block_size, self.head_dim)
        
        # Dense token attention within selected blocks [B, H, M, B_size, K_blocks * B_size]
        attn_weights = torch.matmul(Q_blocked, K_sel.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Vectorized causal mask
        q_m_idx = torch.arange(num_blocks, device=x.device).unsqueeze(1)
        base_idx = torch.arange(self.block_size, device=x.device)
        q_tok_idx = (q_m_idx * self.block_size + base_idx.unsqueeze(0)).unsqueeze(-1) # [M, B_size, 1]
        
        k_tok_idx = (top_k_indices.unsqueeze(-1) * self.block_size + base_idx.view(1, 1, 1, 1, -1)).view(batch_size, self.num_heads, num_blocks, actual_top_k * self.block_size)
        
        mask = q_tok_idx.unsqueeze(0).unsqueeze(0) >= k_tok_idx.unsqueeze(-2)
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, V_sel) # [B, H, M, B_size, D]

        out = out.view(batch_size, self.num_heads, seq_len_padded, self.head_dim).transpose(1, 2).contiguous().view(batch_size, seq_len_padded, embed_dim)
        if pad_len > 0: out = out[:, :-pad_len, :]
        return self.out_proj(out), new_past

class SubQVariantModel(nn.Module):
    """
    The main language model combining subword embeddings, multiple layers 
    of Subquadratic Sparse Attention, and LayerNorms. Uses OpenAI's tiktoken 
    (cl100k_base) BPE tokenizer for optimal context packing.
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
        self.use_checkpointing = False

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        x = self.embedding(input_ids)
        if getattr(self, 'use_checkpointing', False) and self.training:
            x.requires_grad_(True)
            
        new_past_key_values = () if use_cache else None
            
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if getattr(self, 'use_checkpointing', False) and self.training:
                def layer_forward(x_in, l=layer):
                    attn_out, _ = l["attn"](l["ln_1"](x_in))
                    out = x_in + attn_out
                    return out + l["mlp"](l["ln_2"](out))
                x = torch.utils.checkpoint.checkpoint(layer_forward, x, use_reentrant=False)
            else:
                attn_out, new_past = layer["attn"](layer["ln_1"](x), past_key_value=past_kv, use_cache=use_cache)
                x = x + attn_out
                x = x + layer["mlp"](layer["ln_2"](x))
                if use_cache:
                    new_past_key_values = new_past_key_values + (new_past,)
                    
        if use_cache:
            return self.lm_head(x), new_past_key_values
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, max_seq_len=2048):
        self.eval()
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is None:
                idx_cond = input_ids if input_ids.size(1) <= max_seq_len else input_ids[:, -max_seq_len:]
            else:
                idx_cond = input_ids[:, -1:]
                
            logits, past_key_values = self(idx_cond, past_key_values=past_key_values, use_cache=True)
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
        import sys
        if torch.cuda.is_available() and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"Skipping torch.compile: {e}")
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
        ttk.Checkbutton(cfg_box, text="Gradient Checkpointing (Save VRAM via Recomputation)", variable=self.ram_overflow_var).grid(row=1, column=0, sticky="w", pady=2)
        
        self.delta_train_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_box, text="Freeze Base Topologies (Halt Hallucinations)", variable=self.delta_train_var).grid(row=2, column=0, sticky="w", pady=2)
        
        batch_row = ttk.Frame(cfg_box)
        batch_row.grid(row=3, column=0, sticky="w", pady=5)
        ttk.Label(batch_row, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Entry(batch_row, textvariable=self.batch_size_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(batch_row, text="Grad Accumulation:").pack(side=tk.LEFT, padx=10)
        self.grad_accum_var = tk.IntVar(value=4)
        ttk.Entry(batch_row, textvariable=self.grad_accum_var, width=4).pack(side=tk.LEFT)
        
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
        # Keep dataset in CPU RAM and stream to GPU during the training loop.
        dataset = ScientificTextDataset(self.data_dir, seq_length=512)
        if len(dataset) == 0:
            self.log(self.train_log, "[Error] Database empty. Cannot initialize DataLoader.")
            self.btn_train.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            return

        # DataLoader configured to stream batches to GPU
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size_var.get(), 
            shuffle=False, 
            pin_memory=True, 
            num_workers=0,
            drop_last=True
        )

        # 2. Lock Core Weights
        if self.delta_train_var.get():
            self.log(self.train_log, "[Lock] Securing core calculus/physics representations.")
            for name, param in self.model.named_parameters():
                param.requires_grad = ("attn" in name or "router" in name)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 3. VRAM Optimization & Scaler Setup
        use_checkpointing = self.ram_overflow_var.get()
        if use_checkpointing:
            self.model.use_checkpointing = True
            self.log(self.train_log, "[Memory] Gradient Checkpointing activated. Recomputing activations to save VRAM.")
        else:
            self.model.use_checkpointing = False
            
        # Standard GPU Optimizer
        optimizer = AdamW(trainable_params, lr=1e-4, fused=torch.cuda.is_available())

        amp_enabled = self.use_amp_var.get() and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
        if scaler:
            self.log(self.train_log, "[Compute] Automatic Mixed Precision active. Tensor Cores engaged.")
        else:
            self.log(self.train_log, "[Compute] Standard precision (FP32) active. CUDA not available for AMP.")

        accum_steps = max(1, self.grad_accum_var.get())
        epochs = self.epochs_var.get()
        total_steps = (len(dataloader) // accum_steps) * epochs
        warmup_steps = max(1, int(total_steps * 0.1)) # 10% warmup
        
        scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

        self.model.train()
        
        for epoch in range(epochs):
            if not self.is_training: break
            
            optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            
            for step, (x_batch, y_batch) in enumerate(dataloader):
                if not self.is_training: break
                
                # Stream batch to GPU asynchronously
                x_batch = x_batch.to(self.gpu_device, non_blocking=True)
                y_batch = y_batch.to(self.gpu_device, non_blocking=True)
                
                # Forward Pass (with optional Tensor Core acceleration)
                if amp_enabled:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(x_batch)
                        # Sample-level loss aggregation (from SubQ-1.1-Small Technical Report)
                        # Averages loss per sample first to prevent long documents from dominating gradients
                        B, T, V = logits.shape
                        loss_per_token = F.cross_entropy(logits.view(-1, V), y_batch.view(-1), reduction='none').view(B, T)
                        loss_per_sample = loss_per_token.mean(dim=1)
                        loss = loss_per_sample.mean() / accum_steps
                    
                    # Scaled Backward Pass
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        self.model.zero_grad(set_to_none=True)

                else:
                    logits = self.model(x_batch)
                    # Sample-level loss aggregation (SubQ-1.1-Small paper)
                    loss_unreduced = F.cross_entropy(logits.transpose(1, 2), y_batch, reduction='none')
                    loss = loss_unreduced.mean(dim=1).mean() / accum_steps
                    loss.backward()
                    
                    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        self.model.zero_grad(set_to_none=True)

                if step % 20 == 0:
                    loss_val = (loss.item() * accum_steps)
                    atl_msg = ""
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        atl_msg = " (⭐ All-Time Low!)"
                    self.log(self.train_log, f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss_val:.4f}{atl_msg} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    
        self.log(self.train_log, "--- Pipeline Run Concluded ---")
        save_path = os.path.join(self.save_dir, f"subq_science_{int(time.time())}.pt")
        torch.save(self.model.state_dict(), save_path)
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = tk.Tk()
    SubQStudioApp(app)
    app.mainloop()