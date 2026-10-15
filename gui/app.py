"""
Puzzle Generator GUI
====================
Antarmuka grafis berbasis tkinter untuk Puzzle Generator.
Memungkinkan pengguna memuat gambar, mengatur parameter grid,
dan menghasilkan potongan puzzle secara visual.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import random
import os

from puzzle_generator.image_loader import load_image, get_image_info, validate_image
from puzzle_generator.piece_cutter import PuzzleConfig, calculate_config, cut_pieces
from puzzle_generator.frame_generator import generate_frame
from puzzle_generator.augmentation import apply_random_rotation, shuffle_pieces
from puzzle_generator.utils import resize_to_fit


class PuzzleGeneratorApp(tk.Tk):
    """Main application window for Puzzle Generator."""
    
    def __init__(self):
        super().__init__()
        
        self.title("🧩 Puzzle World — Generator")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(bg="#2b2b2b")
        
        # State variables
        self.source_image = None         # Original image (numpy RGB)
        self.frame_image = None          # Generated frame (numpy RGB)
        self.pieces = []                 # List of piece dicts
        self.edge_config = None          # Jigsaw edge config
        self.config = None               # PuzzleConfig
        
        # PhotoImage references (prevent garbage collection)
        self._photo_refs = []
        
        # Build the UI
        self._build_toolbar()
        self._build_display_area()
        self._build_status_bar()
    
    # ─────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────
    
    def _build_toolbar(self):
        """Build the top toolbar with controls."""
        toolbar = tk.Frame(self, bg="#3c3f41", padx=10, pady=8)
        toolbar.pack(fill=tk.X)
        
        # Load Image button
        btn_load = tk.Button(
            toolbar,
            text="📂 Load Image",
            command=self._on_load_image,
            bg="#4a8ec2",
            fg="white",
            font=("Helvetica", 11, "bold"),
            padx=12,
            pady=4,
            relief=tk.FLAT,
            cursor="hand2",
        )
        btn_load.pack(side=tk.LEFT, padx=(0, 15))
        
        # Grid Size
        tk.Label(
            toolbar,
            text="Grid Size (M):",
            bg="#3c3f41",
            fg="white",
            font=("Helvetica", 11),
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.grid_var = tk.IntVar(value=3)
        grid_spin = tk.Spinbox(
            toolbar,
            from_=2,
            to=10,
            textvariable=self.grid_var,
            width=4,
            font=("Helvetica", 11),
            justify=tk.CENTER,
        )
        grid_spin.pack(side=tk.LEFT, padx=(0, 15))
        
        # Jigsaw checkbox
        self.jigsaw_var = tk.BooleanVar(value=True)
        chk_jigsaw = tk.Checkbutton(
            toolbar,
            text="✂️ Jigsaw",
            variable=self.jigsaw_var,
            bg="#3c3f41",
            fg="white",
            selectcolor="#3c3f41",
            activebackground="#3c3f41",
            activeforeground="white",
            font=("Helvetica", 11),
        )
        chk_jigsaw.pack(side=tk.LEFT, padx=(0, 10))
        
        # Rotation checkbox
        self.rotation_var = tk.BooleanVar(value=True)
        chk_rotation = tk.Checkbutton(
            toolbar,
            text="🔄 Rotation",
            variable=self.rotation_var,
            bg="#3c3f41",
            fg="white",
            selectcolor="#3c3f41",
            activebackground="#3c3f41",
            activeforeground="white",
            font=("Helvetica", 11),
        )
        chk_rotation.pack(side=tk.LEFT, padx=(0, 15))
        
        # Generate button
        btn_generate = tk.Button(
            toolbar,
            text="▶ Generate Puzzle",
            command=self._on_generate,
            bg="#5ba85a",
            fg="white",
            font=("Helvetica", 11, "bold"),
            padx=15,
            pady=4,
            relief=tk.FLAT,
            cursor="hand2",
        )
        btn_generate.pack(side=tk.LEFT, padx=(0, 10))
    
    def _build_display_area(self):
        """Build the main display area with image previews."""
        # Main container
        main_frame = tk.Frame(self, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # --- Top row: Original Image + Frame ---
        top_frame = tk.Frame(main_frame, bg="#2b2b2b")
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image panel
        left_panel = tk.LabelFrame(
            top_frame,
            text=" 🖼️ Gambar Asli ",
            bg="#2b2b2b",
            fg="#aaaaaa",
            font=("Helvetica", 10, "bold"),
            padx=5,
            pady=5,
        )
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas_original = tk.Canvas(
            left_panel, bg="#1e1e1e", highlightthickness=0
        )
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Frame panel
        right_panel = tk.LabelFrame(
            top_frame,
            text=" 🖼️ Bingkai (Frame) ",
            bg="#2b2b2b",
            fg="#aaaaaa",
            font=("Helvetica", 10, "bold"),
            padx=5,
            pady=5,
        )
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.canvas_frame = tk.Canvas(
            right_panel, bg="#1e1e1e", highlightthickness=0
        )
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Bottom row: Puzzle Pieces ---
        bottom_panel = tk.LabelFrame(
            main_frame,
            text=" 🧩 Potongan Puzzle (Shuffled) ",
            bg="#2b2b2b",
            fg="#aaaaaa",
            font=("Helvetica", 10, "bold"),
            padx=5,
            pady=5,
        )
        bottom_panel.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Scrollable canvas for pieces
        pieces_container = tk.Frame(bottom_panel, bg="#1e1e1e")
        pieces_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_pieces = tk.Canvas(
            pieces_container, bg="#1e1e1e", highlightthickness=0
        )
        
        scrollbar_y = tk.Scrollbar(
            pieces_container, orient=tk.VERTICAL, command=self.canvas_pieces.yview
        )
        scrollbar_x = tk.Scrollbar(
            pieces_container, orient=tk.HORIZONTAL, command=self.canvas_pieces.xview
        )
        
        self.canvas_pieces.configure(
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
        )
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_pieces.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for piece grid
        self.pieces_inner_frame = tk.Frame(self.canvas_pieces, bg="#1e1e1e")
        self.canvas_pieces.create_window(
            (0, 0), window=self.pieces_inner_frame, anchor=tk.NW
        )
        self.pieces_inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas_pieces.configure(
                scrollregion=self.canvas_pieces.bbox("all")
            ),
        )
    
    def _build_status_bar(self):
        """Build the bottom status bar."""
        self.status_var = tk.StringVar(value="Ready — Load an image to begin.")
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            bg="#3c3f41",
            fg="#cccccc",
            font=("Helvetica", 10),
            anchor=tk.W,
            padx=10,
            pady=4,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    # ─────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────
    
    def _on_load_image(self):
        """Handle Load Image button click."""
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("All Files", "*.*"),
            ],
        )
        
        if not filepath:
            return
        
        try:
            self.source_image = load_image(filepath)
            is_valid, msg = validate_image(self.source_image)
            if not is_valid:
                messagebox.showerror("Invalid Image", msg)
                self.source_image = None
                return
            
            info = get_image_info(self.source_image)
            filename = os.path.basename(filepath)
            self.status_var.set(
                f"Loaded: {filename} — {info['width']}×{info['height']}px, "
                f"{info['mode']}"
            )
            
            # Display original image
            self._display_image_on_canvas(
                self.source_image, self.canvas_original
            )
            
            # Clear previous results
            self._clear_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def _on_generate(self):
        """Handle Generate Puzzle button click."""
        if self.source_image is None:
            messagebox.showwarning(
                "No Image", "Please load an image first!"
            )
            return
        
        grid_size = self.grid_var.get()
        use_jigsaw = self.jigsaw_var.get()
        use_rotation = self.rotation_var.get()
        
        try:
            self.status_var.set("Generating puzzle...")
            self.update_idletasks()
            
            # Calculate grid config
            self.config = calculate_config(
                self.source_image, grid_size, border_ratio=0.1
            )
            
            # Cut pieces
            self.pieces, self.edge_config = cut_pieces(
                self.source_image,
                self.config,
                jigsaw=use_jigsaw,
            )
            
            # Generate frame
            self.frame_image = generate_frame(
                self.source_image,
                self.config,
                jigsaw=use_jigsaw,
                edge_config=self.edge_config,
            )
            
            # Apply rotation if enabled
            if use_rotation:
                for piece in self.pieces:
                    rotated, angle = apply_random_rotation(piece['image'])
                    piece['image'] = rotated
                    piece['rotation'] = angle
            
            # Shuffle pieces
            self.pieces = shuffle_pieces(self.pieces)
            
            # Display results
            self._display_image_on_canvas(
                self.frame_image, self.canvas_frame
            )
            self._display_pieces()
            
            N = self.config.total_pieces
            self.status_var.set(
                f"✅ Generated {N} pieces ({grid_size}×{grid_size} grid) — "
                f"Jigsaw: {'ON' if use_jigsaw else 'OFF'}, "
                f"Rotation: {'ON' if use_rotation else 'OFF'}"
            )
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Error: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    # ─────────────────────────────────────────────
    # Display Functions
    # ─────────────────────────────────────────────
    
    def _display_image_on_canvas(
        self, image: np.ndarray, canvas: tk.Canvas
    ):
        """Display a numpy image on a tkinter Canvas, fitted to canvas size."""
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            cw, ch = 400, 300
        
        # Resize to fit canvas
        display_img = resize_to_fit(image, cw - 10, ch - 10)
        
        # Convert numpy to PIL to PhotoImage
        if len(display_img.shape) == 2:
            pil_img = Image.fromarray(display_img, mode='L')
        elif display_img.shape[2] == 4:
            pil_img = Image.fromarray(display_img, mode='RGBA')
        else:
            pil_img = Image.fromarray(display_img, mode='RGB')
        
        photo = ImageTk.PhotoImage(pil_img)
        
        # Keep reference to prevent GC
        self._photo_refs.append(photo)
        
        # Display centered on canvas
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
    
    def _display_pieces(self):
        """Display shuffled puzzle pieces in the bottom panel."""
        # Clear previous pieces
        for widget in self.pieces_inner_frame.winfo_children():
            widget.destroy()
        
        if not self.pieces:
            return
        
        M = self.config.grid_size
        
        # Calculate display size for each piece
        # Fit pieces in a grid with M columns
        self.canvas_pieces.update_idletasks()
        canvas_w = self.canvas_pieces.winfo_width()
        if canvas_w < 10:
            canvas_w = 800
        
        # Allow for padding between pieces
        padding = 6
        piece_display_w = max(50, (canvas_w - padding * (M + 1)) // M)
        piece_display_h = piece_display_w  # Keep roughly square
        
        # Cap maximum size
        piece_display_w = min(piece_display_w, 200)
        piece_display_h = min(piece_display_h, 200)
        
        for idx, piece_data in enumerate(self.pieces):
            row = idx // M
            col = idx % M
            
            piece_img = piece_data['image']
            orig_idx = piece_data['original_index']
            rotation = piece_data['rotation']
            
            # Resize piece for display
            display_piece = resize_to_fit(
                piece_img, piece_display_w, piece_display_h
            )
            
            # Convert to PIL → PhotoImage
            if len(display_piece.shape) == 3 and display_piece.shape[2] == 4:
                # RGBA: composite onto a background color
                pil_img = Image.fromarray(display_piece, mode='RGBA')
                bg = Image.new('RGBA', pil_img.size, (40, 40, 40, 255))
                composite = Image.alpha_composite(bg, pil_img)
                pil_img = composite.convert('RGB')
            elif len(display_piece.shape) == 2:
                pil_img = Image.fromarray(display_piece, mode='L')
            else:
                pil_img = Image.fromarray(display_piece, mode='RGB')
            
            photo = ImageTk.PhotoImage(pil_img)
            self._photo_refs.append(photo)
            
            # Create piece frame with label
            piece_frame = tk.Frame(
                self.pieces_inner_frame, bg="#1e1e1e", padx=2, pady=2
            )
            piece_frame.grid(
                row=row, column=col, padx=padding // 2, pady=padding // 2
            )
            
            lbl_img = tk.Label(
                piece_frame,
                image=photo,
                bg="#282828",
                relief=tk.RAISED,
                borderwidth=1,
            )
            lbl_img.pack()
            
            # Label showing original index and rotation
            label_text = f"#{orig_idx + 1}"
            if rotation != 0:
                label_text += f" ↻{rotation}°"
            
            lbl_text = tk.Label(
                piece_frame,
                text=label_text,
                bg="#1e1e1e",
                fg="#888888",
                font=("Helvetica", 8),
            )
            lbl_text.pack()
    
    def _clear_results(self):
        """Clear previous puzzle results from display."""
        self.frame_image = None
        self.pieces = []
        self.edge_config = None
        self.config = None
        self._photo_refs.clear()
        
        self.canvas_frame.delete("all")
        for widget in self.pieces_inner_frame.winfo_children():
            widget.destroy()
        
        # Re-display original if loaded
        if self.source_image is not None:
            self._display_image_on_canvas(
                self.source_image, self.canvas_original
            )


def run_app():
    """Launch the Puzzle Generator GUI application."""
    app = PuzzleGeneratorApp()
    app.mainloop()
