import os
import re
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import arxiv
from pypdf import PdfReader
import urllib.request
import fitz
from PIL import Image, ImageTk
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
import concurrent.futures

class SubQHarvesterApp:
    """
    The primary GUI application for the SubQ Harvester.
    Orchestrates the pipeline for fetching, parsing, and cleaning scientific 
    textbooks, arXiv papers, and open web PDFs. Ensures text is properly 
    formatted with `<sep>` boundary tokens for long-context packing.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("SubQ Harvester: Automated Data Pipeline")
        self.root.geometry("800x600")
        
        # Ensure the output directory matches the SubQ V4 Studio target
        self.output_dir = "./science_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.chunk_lock = threading.Lock()
        self.current_chunk_index = 1
        self.current_chunk_size = 0
        self.max_chunk_size = 50 * 1024 * 1024 # 50 MB
        self._init_chunk_state()
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.build_arxiv_tab()
        self.build_pdf_tab()
        self.build_web_pdf_tab()
        self.build_auto_expert_tab()

    def log(self, text_widget, msg):
        text_widget.insert(tk.END, msg + "\n")
        text_widget.see(tk.END)

    def _init_chunk_state(self):
        import glob
        existing_chunks = glob.glob(os.path.join(self.output_dir, "expert_chunk_*.txt"))
        if existing_chunks:
            indices = []
            for f in existing_chunks:
                match = re.search(r'expert_chunk_(\d+)\.txt', f)
                if match: indices.append(int(match.group(1)))
            if indices:
                self.current_chunk_index = max(indices)
                last_chunk = os.path.join(self.output_dir, f"expert_chunk_{self.current_chunk_index:03d}.txt")
                if os.path.exists(last_chunk):
                    self.current_chunk_size = os.path.getsize(last_chunk)
                    if self.current_chunk_size >= self.max_chunk_size:
                        self.current_chunk_index += 1
                        self.current_chunk_size = 0

    # ==========================================
    # TEXT CLEANING UTILITY
    # ==========================================
    def clean_and_save_text(self, raw_text, filename, chunk_mode=False):
        """
        Formats the text to be easily digestible by the SubQ Tokenizer.
        Removes strange PDF formatting artifacts and enforces a rigid character set.
        Crucially, it appends a `<sep>` token at the end of the document to prevent 
        the model from hallucinating cross-document relationships when sequences are 
        packed tightly during training.
        """
        # Remove multiple newlines, weird PDF artifact spacing, and unprintable chars
        clean_text = re.sub(r'\n+', '\n', raw_text)
        clean_text = re.sub(r'[^\x00-\x7Fα-ωΑ-Ω∑∫∂∇∈⊂≈≠≡≤≥∞⊗⊕ℏ]', ' ', clean_text) 
        
        # Add document separator as mentioned in the SubQ-1.1-Small paper
        clean_text = clean_text.strip() + "\n<sep>\n"
        
        if chunk_mode:
            with self.chunk_lock:
                if self.current_chunk_size >= self.max_chunk_size:
                    self.current_chunk_index += 1
                    self.current_chunk_size = 0
                    
                chunk_filename = f"expert_chunk_{self.current_chunk_index:03d}.txt"
                filepath = os.path.join(self.output_dir, chunk_filename)
                
                encoded_text = clean_text.encode('utf-8')
                with open(filepath, 'ab') as f:
                    f.write(encoded_text)
                self.current_chunk_size += len(encoded_text)
                return filepath
        else:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            return filepath

    # ==========================================
    # TAB 1: ARXIV API FETCHER
    # ==========================================
    def build_arxiv_tab(self):
        arxiv_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(arxiv_frame, text="🌐 arXiv Research Fetcher")
        
        desc = (
            "Search Cornell University's arXiv database for cutting-edge physics, "
            "math, and computer science papers. The app will download the PDFs, "
            "parse the raw text, and save them as formatted .txt files."
        )
        ttk.Label(arxiv_frame, text=desc, wraplength=750).pack(anchor="w", pady=(0, 10))
        
        input_row = ttk.Frame(arxiv_frame)
        input_row.pack(fill='x', pady=5)
        
        ttk.Label(input_row, text="Search Query (e.g., 'Tensor Calculus'):").pack(side=tk.LEFT)
        self.query_var = tk.StringVar()
        ttk.Entry(input_row, textvariable=self.query_var, width=40).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(input_row, text="Max Papers:").pack(side=tk.LEFT)
        self.max_results_var = tk.IntVar(value=5)
        ttk.Entry(input_row, textvariable=self.max_results_var, width=5).pack(side=tk.LEFT, padx=10)
        
        self.btn_fetch = ttk.Button(arxiv_frame, text="Fetch & Parse Papers", command=self.start_arxiv_fetch)
        self.btn_fetch.pack(anchor="w", pady=10)
        
        self.arxiv_log = scrolledtext.ScrolledText(arxiv_frame, height=20, bg="#1e1e1e", fg="#4da6ff")
        self.arxiv_log.pack(fill="both", expand=True)

    def start_arxiv_fetch(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query.")
            return
            
        self.btn_fetch.config(state=tk.DISABLED)
        threading.Thread(target=self.fetch_arxiv_logic, args=(query,), daemon=True).start()

    def fetch_arxiv_logic(self, query):
        self.log(self.arxiv_log, f"--- Initiating API connection to arXiv for: '{query}' ---")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.max_results_var.get(),
            sort_by=arxiv.SortCriterion.Relevance
        )

        try:
            for paper in client.results(search):
                self.log(self.arxiv_log, f"\nFound: {paper.title}")
                self.log(self.arxiv_log, "Downloading PDF...")
                
                # Create temporary PDF file
                safe_id = paper.get_short_id().replace('/', '_')
                temp_pdf = f"temp_{safe_id}.pdf"
                import urllib.request
                urllib.request.urlretrieve(paper.pdf_url, temp_pdf)
                
                self.log(self.arxiv_log, "Extracting text matrices...")
                raw_text = f"TITLE: {paper.title}\nAUTHORS: {[a.name for a in paper.authors]}\nABSTRACT: {paper.summary}\n\n"
                
                # Parse the downloaded PDF
                try:
                    reader = PdfReader(temp_pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            raw_text += extracted + "\n"
                            
                    # Clean and Save
                    safe_title = re.sub(r'[^a-zA-Z0-9]', '_', paper.title)[:50]
                    txt_filename = f"arxiv_{safe_title}.txt"
                    saved_path = self.clean_and_save_text(raw_text, txt_filename)
                    
                    self.log(self.arxiv_log, f"Successfully parsed and saved to: {saved_path}")
                except Exception as e:
                    self.log(self.arxiv_log, f"Error parsing PDF for {paper.title}: {e}")
                finally:
                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf) # Clean up temp PDF
                        
            self.log(self.arxiv_log, "\n--- arXiv Fetch Complete! Data is ready for SubQ Studio. ---")
        except Exception as e:
            self.log(self.arxiv_log, f"\nAPI Error: {e}")
            
        self.root.after(0, lambda: self.btn_fetch.config(state=tk.NORMAL))

    # ==========================================
    # TAB 3: WEB PDF FETCHER
    # ==========================================
    def build_web_pdf_tab(self):
        main_pane = ttk.PanedWindow(self.notebook, orient=tk.HORIZONTAL)
        self.notebook.add(main_pane, text="🌐 Web PDF Fetcher")
        
        web_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(web_frame, weight=1)
        
        viewer_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(viewer_frame, weight=1)
        
        # --- Left Pane (web_frame) ---
        desc = (
            "Search the open web for textbook PDFs and download them directly into the "
            "data pipeline. Uses duckduckgo syntax (ext:pdf)."
        )
        ttk.Label(web_frame, text=desc, wraplength=400).pack(anchor="w", pady=(0, 10))
        
        input_row = ttk.Frame(web_frame)
        input_row.pack(fill='x', pady=5)
        
        ttk.Label(input_row, text="Topic:").pack(side=tk.LEFT)
        self.web_query_var = tk.StringVar()
        ttk.Entry(input_row, textvariable=self.web_query_var, width=30).pack(side=tk.LEFT, padx=5)
        
        self.btn_web_search = ttk.Button(input_row, text="Search Web", command=self.start_web_search)
        self.btn_web_search.pack(side=tk.LEFT)
        
        # Results Treeview
        columns = ("title", "url")
        self.web_tree = ttk.Treeview(web_frame, columns=columns, show="headings", height=8)
        self.web_tree.heading("title", text="Document Title")
        self.web_tree.heading("url", text="PDF URL")
        self.web_tree.column("title", width=200)
        self.web_tree.column("url", width=200)
        self.web_tree.pack(fill="x", pady=10)
        
        self.btn_web_download = ttk.Button(web_frame, text="Download & Parse Selected", command=self.download_selected_web_pdf, state=tk.DISABLED)
        self.btn_web_download.pack(anchor="w", pady=5)
        
        self.web_tree.bind("<<TreeviewSelect>>", self.on_web_tree_select)
        
        self.web_log = scrolledtext.ScrolledText(web_frame, height=12, bg="#1e1e1e", fg="#ffcc00")
        self.web_log.pack(fill="both", expand=True)

        # --- Right Pane (viewer_frame) ---
        ttk.Label(viewer_frame, text="Live PDF Preview", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0,5))
        
        self.preview_canvas = tk.Canvas(viewer_frame, bg="gray", width=400, height=500)
        self.preview_canvas.pack(fill="both", expand=True)
        
        ctrl_frame = ttk.Frame(viewer_frame)
        ctrl_frame.pack(fill="x", pady=5)
        
        self.btn_prev_page = ttk.Button(ctrl_frame, text="<< Prev", command=self.preview_prev_page, state=tk.DISABLED)
        self.btn_prev_page.pack(side=tk.LEFT)
        
        self.lbl_page_num = ttk.Label(ctrl_frame, text="Page 0 / 0")
        self.lbl_page_num.pack(side=tk.LEFT, padx=10)
        
        self.btn_next_page = ttk.Button(ctrl_frame, text="Next >>", command=self.preview_next_page, state=tk.DISABLED)
        self.btn_next_page.pack(side=tk.LEFT)
        
        self.current_preview_doc = None
        self.current_preview_page = 0
        self.current_preview_path = None
        self.preview_image = None

    def start_web_search(self):
        query = self.web_query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a topic to search.")
            return
            
        if DDGS is None:
            messagebox.showerror("Error", "ddgs module is missing.")
            return

        self.btn_web_search.config(state=tk.DISABLED)
        self.web_tree.delete(*self.web_tree.get_children())
        self.log(self.web_log, f"--- Searching Web for: {query} ext:pdf ---")
        threading.Thread(target=self.web_search_logic, args=(query,), daemon=True).start()

    def web_search_logic(self, query):
        try:
            full_query = f"{query} ext:pdf"
            results = DDGS().text(full_query, max_results=15)
            
            if not results:
                self.log(self.web_log, "No PDF results found.")
            else:
                count = 0
                for res in results:
                    title = res.get('title', 'Unknown Title')
                    url = res.get('href', '')
                    if '.pdf' in url.lower():
                        self.web_tree.insert("", tk.END, values=(title, url))
                        count += 1
                
                if count == 0:
                    self.log(self.web_log, "Found links, but none appeared to be direct PDFs.")
                else:
                    self.log(self.web_log, f"Search complete. Found {count} PDFs. Select a document to download.")
                
        except Exception as e:
            self.log(self.web_log, f"Search Error: {e}")
            
        self.root.after(0, lambda: self.btn_web_search.config(state=tk.NORMAL))

    def on_web_tree_select(self, event):
        selected = self.web_tree.selection()
        if not selected:
            self.btn_web_download.config(state=tk.DISABLED)
            return
            
        self.btn_web_download.config(state=tk.NORMAL)
        item = self.web_tree.item(selected[0])
        title, url = item['values']
        
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(200, 250, text="Downloading Preview...", fill="white", font=("Helvetica", 14))
        
        self.btn_prev_page.config(state=tk.DISABLED)
        self.btn_next_page.config(state=tk.DISABLED)
        self.lbl_page_num.config(text="Page 0 / 0")
        
        if self.current_preview_doc:
            self.current_preview_doc.close()
            self.current_preview_doc = None
            
        threading.Thread(target=self.fetch_preview_logic, args=(title, url), daemon=True).start()

    def fetch_preview_logic(self, title, url):
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:50]
        temp_pdf = f"temp_preview_{safe_title}.pdf"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_pdf, 'wb') as out_file:
                out_file.write(response.read())
                
            self.current_preview_path = temp_pdf
            self.current_preview_doc = fitz.open(temp_pdf)
            self.current_preview_page = 0
            
            self.root.after(0, self.render_preview_page)
            
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.show_preview_error(err))

    def show_preview_error(self, err_msg):
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(200, 250, text=f"Preview Error:\n{err_msg}", fill="red", width=380)

    def render_preview_page(self):
        if not self.current_preview_doc: return
        
        total_pages = len(self.current_preview_doc)
        page = self.current_preview_doc.load_page(self.current_preview_page)
        
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        self.preview_canvas.update_idletasks()
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        if canvas_w < 10: canvas_w = 400
        if canvas_h < 10: canvas_h = 500
        
        img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(img)
        
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w//2, canvas_h//2, image=self.preview_image, anchor="center")
        
        self.lbl_page_num.config(text=f"Page {self.current_preview_page + 1} / {total_pages}")
        self.btn_prev_page.config(state=tk.NORMAL if self.current_preview_page > 0 else tk.DISABLED)
        self.btn_next_page.config(state=tk.NORMAL if self.current_preview_page < total_pages - 1 else tk.DISABLED)

    def preview_prev_page(self):
        if self.current_preview_doc and self.current_preview_page > 0:
            self.current_preview_page -= 1
            self.render_preview_page()

    def preview_next_page(self):
        if self.current_preview_doc and self.current_preview_page < len(self.current_preview_doc) - 1:
            self.current_preview_page += 1
            self.render_preview_page()

    def download_selected_web_pdf(self):
        selected = self.web_tree.selection()
        if not selected:
            return
            
        item = self.web_tree.item(selected[0])
        title, url = item['values']
        
        if not self.current_preview_path or not os.path.exists(self.current_preview_path):
            messagebox.showwarning("Warning", "Preview is still downloading. Please wait.")
            return
        
        self.btn_web_download.config(state=tk.DISABLED)
        threading.Thread(target=self.web_parse_logic, args=(title, self.current_preview_path), daemon=True).start()

    def web_parse_logic(self, title, temp_pdf):
        self.log(self.web_log, f"\nParsing Cached PDF: {title}")
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:50]
        
        try:
            reader = PdfReader(temp_pdf)
            raw_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"
                    
            txt_filename = f"web_{safe_title}.txt"
            saved_path = self.clean_and_save_text(raw_text, txt_filename)
            self.log(self.web_log, f"Successfully parsed and saved to: {saved_path}")
            
        except Exception as e:
            self.log(self.web_log, f"Error parsing {title}: {e}")
        finally:
            self.root.after(0, lambda: self.btn_web_download.config(state=tk.NORMAL if self.web_tree.selection() else tk.DISABLED))

    # ==========================================
    # TAB 4: LOCAL TEXTBOOK PARSER
    # ==========================================
    def build_pdf_tab(self):
        pdf_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(pdf_frame, text="📖 Local Textbook Parser")
        
        desc = (
            "Have a large PDF textbook on Quantum Mechanics or Calculus? "
            "Select it here. The engine will strip the visual formatting and extract "
            "the raw text strings, saving it directly into your training database."
        )
        ttk.Label(pdf_frame, text=desc, wraplength=750).pack(anchor="w", pady=(0, 10))
        
        self.btn_browse = ttk.Button(pdf_frame, text="Select PDF Textbook", command=self.parse_local_pdf)
        self.btn_browse.pack(anchor="w", pady=10)
        
        self.pdf_log = scrolledtext.ScrolledText(pdf_frame, height=20, bg="#1e1e1e", fg="#00ffcc")
        self.pdf_log.pack(fill="both", expand=True)

    def parse_local_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return
            
        self.btn_browse.config(state=tk.DISABLED)
        threading.Thread(target=self.local_pdf_logic, args=(file_path,), daemon=True).start()

    def local_pdf_logic(self, file_path):
        filename = os.path.basename(file_path)
        self.log(self.pdf_log, f"--- Analyzing Textbook: {filename} ---")
        
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            self.log(self.pdf_log, f"Detected {total_pages} pages. Initiating extraction...")
            
            raw_text = ""
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"
                    
                if i % 50 == 0 and i > 0:
                    self.log(self.pdf_log, f"Parsed {i}/{total_pages} pages...")
                    
            safe_name = os.path.splitext(filename)[0]
            txt_filename = f"textbook_{safe_name}.txt"
            saved_path = self.clean_and_save_text(raw_text, txt_filename)
            
            self.log(self.pdf_log, f"Extraction Complete! Result formatted and saved to:\n{saved_path}")
        except Exception as e:
            self.log(self.pdf_log, f"Failed to parse textbook: {e}")
            
        self.root.after(0, lambda: self.btn_browse.config(state=tk.NORMAL))

    # ==========================================
    # TAB 5: AUTO-EXPERT PIPELINE
    # ==========================================
    def build_auto_expert_tab(self):
        expert_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(expert_frame, text="🤖 Auto-Expert Pipeline")
        
        desc = (
            "Automatically harvest the absolute cutting-edge of arXiv papers across "
            "Quantum Mechanics, High Energy Physics, and foundational mathematics "
            "to build a domain-expert dataset. Runs silently in the background."
        )
        ttk.Label(expert_frame, text=desc, wraplength=750).pack(anchor="w", pady=(0, 10))
        
        input_row = ttk.Frame(expert_frame)
        input_row.pack(fill='x', pady=5)
        
        ttk.Label(input_row, text="Papers per field:").pack(side=tk.LEFT)
        self.expert_max_var = tk.IntVar(value=10)
        ttk.Entry(input_row, textvariable=self.expert_max_var, width=5).pack(side=tk.LEFT, padx=10)
        
        self.btn_expert = ttk.Button(expert_frame, text="Initiate Expert Harvesting", command=self.start_expert_harvest)
        self.btn_expert.pack(anchor="w", pady=10)
        
        self.expert_progress = ttk.Progressbar(expert_frame, orient="horizontal", length=400, mode="determinate")
        self.expert_progress.pack(fill="x", pady=5)
        
        self.expert_log = scrolledtext.ScrolledText(expert_frame, height=20, bg="#1e1e1e", fg="#ff4444")
        self.expert_log.pack(fill="both", expand=True)

    def start_expert_harvest(self):
        self.btn_expert.config(state=tk.DISABLED)
        threading.Thread(target=self.expert_harvest_logic, daemon=True).start()

    def expert_harvest_logic(self):
        cutting_edge_queries = [
            ("Quantum Physics", "cat:quant-ph"),
            ("High Energy Physics (Theory)", "cat:hep-th"),
            ("High Energy Physics (Phenomenology)", "cat:hep-ph"),
            ("Quantum Chromodynamics", "all:\"quantum chromodynamics\""),
            ("Mathematical Physics", "cat:math-ph")
        ]
        
        foundational_queries = [
            ("Classical Mechanics", "cat:physics.class-ph"),
            ("General Physics Fundamentals", "cat:physics.gen-ph"),
            ("Thermodynamics", "all:\"thermodynamics\""),
            ("Electromagnetism", "all:\"electromagnetism\""),
            ("Software Engineering (Code)", "cat:cs.SE")
        ]
        
        max_papers = self.expert_max_var.get()
        client = arxiv.Client()
        
        self.log(self.expert_log, "=== PHASE 1: CUTTING-EDGE QUANTUM ===")
        for name, query in cutting_edge_queries:
            self.log(self.expert_log, f"\n--- Harvesting: {name} ---")
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            self._execute_harvest_search(client, search, "expert")
            
        self.log(self.expert_log, "\n=== PHASE 2: FOUNDATIONAL BASICS ===")
        for name, query in foundational_queries:
            self.log(self.expert_log, f"\n--- Harvesting: {name} ---")
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            self._execute_harvest_search(client, search, "basics")
                
        self.log(self.expert_log, "\n--- EXPERT HARVEST COMPLETE ---")
        self.root.after(0, lambda: self.btn_expert.config(state=tk.NORMAL))

    def _process_single_paper(self, paper, prefix):
        import urllib.request
        try:
            safe_id = paper.get_short_id().replace('/', '_')
            temp_pdf = f"temp_{prefix}_{safe_id}.pdf"
            
            try:
                urllib.request.urlretrieve(paper.pdf_url, temp_pdf)
                
                raw_text = f"TITLE: {paper.title}\nAUTHORS: {[a.name for a in paper.authors]}\nABSTRACT: {paper.summary}\n\n"
                reader = PdfReader(temp_pdf)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        raw_text += extracted + "\n"
                        
                # Notice we pass empty string for filename, since chunk_mode handles it
                self.clean_and_save_text(raw_text, "", chunk_mode=True)
                return True
                
            except Exception as e:
                # Silently catch parsing errors for massive datasets
                return False
            finally:
                if os.path.exists(temp_pdf):
                    os.remove(temp_pdf)
                    
        except Exception as e:
            return False

    def _execute_harvest_search(self, client, search, prefix):
        papers = list(client.results(search))
        total = len(papers)
        if total == 0:
            return
            
        self.root.after(0, lambda: self.expert_progress.config(maximum=total, value=0))
        
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._process_single_paper, p, prefix) for p in papers]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if future.result():
                    success_count += 1
                    
                # Update progress bar safely
                self.root.after(0, lambda v=i+1: self.expert_progress.config(value=v))
                
                # Log milestone every 10% or 50 papers to avoid freezing UI
                if (i + 1) % max(1, total // 10) == 0 or (i + 1) % 50 == 0:
                    self.log(self.expert_log, f"Progress: {i+1}/{total} papers processed...")
                    
        self.log(self.expert_log, f"Completed section. Successfully integrated {success_count}/{total} papers into Chunk DB.")

if __name__ == "__main__":
    app = tk.Tk()
    SubQHarvesterApp(app)
    app.mainloop()