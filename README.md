# SubQ Studio & Harvester 🌌

(CURRENTLY PROOF OF CONCEPT/ ROUGH INTEGRATION) An end-to-end, open-source pipeline for scraping scientific knowledge (arXiv, PDFs, web search) and training a highly efficient **Subquadratic Sparse Attention (SSA)** language model locally on consumer hardware (e.g., RTX 4070).

Built specifically to implement methodologies from the **SubQ-1.1-Small** technical report, this project features linear-scaling sparse attention mechanics with Rotary Positional Embeddings (RoPE) for stable long-context reasoning.

## Components

### 1. SubQ Harvester (`subqharvester.py`)
An automated dataset generation pipeline with a Tkinter GUI.
- **Auto-Expert Pipeline**: Executes targeted arXiv queries across Physics, Math, and Software Engineering.
- **Local PDF Parser**: Strips visual formatting and extracts raw text from dense textbooks.
- **Document Boundaries**: Automatically injects `<sep>` tokens between documents to prevent cross-document hallucination during long-context packing.

### 2. SubQ Studio (`subqstudio.py`)
A local training and active inference studio for the SubQ architecture.
- **Subquadratic Sparse Attention**: Employs content-dependent block-level routing for near-linear compute scaling at long contexts.
- **Rotary Positional Embeddings (RoPE)**: Replaces traditional absolute position embeddings, allowing the model to smoothly extrapolate and generate beyond its training sequence length.
- **Sample-Level Loss Aggregation**: Stabilizes gradient updates across packed long-context examples by averaging loss over the time dimension.
- **Hardware Optimized**: Features a streaming PyTorch `DataLoader` with host RAM optimizer offloading, allowing you to train multi-million parameter models on standard 12GB VRAM GPUs without OOM errors.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Harvest Data**: 
   Run `python subqharvester.py` to open the GUI. Run the Auto-Expert Pipeline or manually add PDFs to build a chunked training corpus in `./science_data/`.
2. **Train the Model**: 
   Run `python subqstudio.py` and switch to the `CUDA Core Engine` tab. Ensure your parameters are set and hit "Ignite Training Pipeline".
3. **Inference**: 
   Once a `.pt` model is saved in `./subq_models/`, navigate to the `Active Inference` tab, load your model, and start prompting!
