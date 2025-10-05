# nasa_explorer.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import math
import requests
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CLASES AUXILIARES
# =============================================================================

class NASADataFetcher:
    def __init__(self):
        self.apis = {
            "earth": "https://api.nasa.gov/planetary/earth/imagery",
            "mars": "https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos",
            "apod": "https://api.nasa.gov/planetary/apod",
            "library": "https://images-api.nasa.gov/search"
        }
        self.api_key = "DEMO_KEY"
        
    def fetch_earth_imagery(self, lat, lon, date=None, dim=0.1):
        """Obtiene imágenes recientes de la Tierra"""
        try:
            params = {'lon': lon, 'lat': lat, 'dim': dim, 'api_key': self.api_key}
            if date:
                params['date'] = date
            
            response = requests.get(self.apis["earth"], params=params, timeout=30)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            return None
        except Exception as e:
            print(f"Error fetching Earth imagery: {e}")
            return None

class GigapixelImageHandler:
    def __init__(self):
        self.tile_cache = {}
        self.max_cache_size = 100
        
    def create_simulated_gigapixel(self, width=10000, height=5000):
        """Crea una imagen simulada de alta resolución para demostración"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Añadir estrellas
        np.random.seed(42)
        n_stars = 50000
        star_positions = np.random.randint(0, [width, height], (n_stars, 2))
        star_sizes = np.random.exponential(0.5, n_stars) + 0.1
        
        for (x, y), size in zip(star_positions, star_sizes):
            if size > 1:
                brightness = min(255, int(size * 80))
                cv2.circle(image, (x, y), int(size), (brightness, brightness, brightness), -1)
        
        # Añadir galaxia espiral simulada
        center_x, center_y = width // 2, height // 2
        for angle in np.linspace(0, 2*np.pi, 1000):
            radius = 800 + 500 * np.sin(angle * 5)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle) * 0.3)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), 2, (200, 150, 100), -1)
        
        return image

class CollaborativeAnnotationSystem:
    def __init__(self):
        self.annotations = {}
        self.annotation_id_counter = 0
        
    def add_annotation(self, image_id, x1, y1, x2, y2, label, user_id="anonymous", feature_type="unknown"):
        annotation_id = f"{image_id}_{self.annotation_id_counter}"
        self.annotation_id_counter += 1
        
        annotation = {
            "id": annotation_id,
            "coords": (x1, y1, x2, y2),
            "label": label,
            "user_id": user_id,
            "feature_type": feature_type,
            "timestamp": datetime.now().isoformat(),
            "verified": False,
        }
        
        self.annotations[annotation_id] = annotation
        return annotation_id

class TemporalAnalysis:
    def __init__(self):
        self.time_series_data = {}
        
    def compare_temporal_images(self, image1, image2):
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        diff = cv2.absdiff(image1, image2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        changes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                changes.append({
                    "bbox": (x, y, w, h),
                    "area": area,
                    "center": (x + w//2, y + h//2)
                })
        
        return {
            "change_mask": thresh,
            "changes_detected": len(changes),
            "change_regions": changes,
            "percent_change": (np.sum(thresh > 0) / (image1.shape[0] * image1.shape[1])) * 100
        }

class MuseumMode:
    def __init__(self, main_app):
        self.app = main_app
        self.kiosk_mode = False
        self.tour_points = []
        
    def enable_kiosk_mode(self):
        self.kiosk_mode = True
        self.app.root.attributes('-fullscreen', True)
        self.app.left_panel.pack_forget()
        self.app.bottom_panel.pack_forget()
        self.show_simple_controls()
    
    def disable_kiosk_mode(self):
        self.kiosk_mode = False
        self.app.root.attributes('-fullscreen', False)
        self.app.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.app.bottom_panel.pack(fill=tk.X, pady=(10, 0))
    
    def show_simple_controls(self):
        control_frame = ttk.Frame(self.app.root)
        control_frame.place(x=20, y=20, width=200, height=150)
        
        ttk.Label(control_frame, text="NASA EXPLORER", 
                 font=('Arial', 14, 'bold')).pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Zoom In", 
                  command=self.app.zoom_in).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Zoom Out", 
                  command=self.app.zoom_out).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Exit Kiosk", 
                  command=self.disable_kiosk_mode).pack(fill=tk.X, pady=2)

# =============================================================================
# CLASE PRINCIPAL NASA IMAGE EXPLORER
# =============================================================================

class NASAImageExplorerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Multi-Mission Image Explorer v2.0")
        self.root.geometry("1400x900")
        
        # Variables principales
        self.images = {}
        self.active_image_id = None
        self.photo = None
        self.canvas_image = None
        self.scale = 1.0
        self.original_size = (0, 0)
        self.original_image = None
        self.cv_image = None
        self.current_filtered_image = None
        
        # Sistemas de filtros y capas
        self.transformaciones = {}
        self.active_filters = []
        
        # Navegación y anotaciones
        self.labels = []
        self.current_label = None
        self.label_mode = False
        self.start_x = None
        self.start_y = None
        self.drag_start_x = None
        self.drag_start_y = None
        self.is_dragging = False
        
        # Sistemas NASA
        self.nasa_fetcher = NASADataFetcher()
        self.gigapixel_handler = GigapixelImageHandler()
        self.collab_annotations = CollaborativeAnnotationSystem()
        self.temporal_analyzer = TemporalAnalysis()
        self.museum_mode = MuseumMode(self)
        
        # Configuración de UI
        self.setup_ui()
        self.load_sample_datasets()
        self.create_demo_datasets()

    def create_demo_datasets(self):
        self.nasa_datasets = {
            "Hubble_Andromeda": {
                "description": "Hubble Andromeda Galaxy 2.5 Gigapixels Simulation",
                "type": "stellar",
                "simulated_size": (10000, 5000),
            },
            "MRO_Mars_Global": {
                "description": "Mars Reconnaissance Orbiter Global Map Simulation", 
                "type": "planetary",
                "simulated_size": (8000, 4000),
            },
            "LRO_Lunar_Mosaic": {
                "description": "Lunar Reconnaissance Orbiter Mosaic Simulation",
                "type": "planetary", 
                "simulated_size": (12000, 6000),
            }
        }

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Panel central
        center_panel = ttk.Frame(main_frame)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_center_panel(center_panel)
        self.setup_bottom_panel(center_panel)

    def setup_left_panel(self, parent):
        # Banner NASA
        banner_frame = ttk.Frame(parent, height=60)
        banner_frame.pack(fill=tk.X, pady=(0, 10))
        banner_frame.pack_propagate(False)
        
        banner_label = ttk.Label(banner_frame, text="🚀 NASA IMAGE EXPLORER", 
                               font=('Arial', 14, 'bold'), justify=tk.CENTER)
        banner_label.pack(expand=True, fill=tk.BOTH)
        
        # Misiones NASA
        mission_frame = ttk.LabelFrame(parent, text="NASA Missions", padding=10)
        mission_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mission_frame, text="Select Mission:").pack(anchor=tk.W)
        self.mission_var = tk.StringVar()
        missions = ["TESS FFI", "Lunar Reconnaissance", "Earth Observatory", "Mars Rover", 
                   "Hubble_Andromeda", "MRO_Mars_Global", "LRO_Lunar_Mosaic"]
        mission_combo = ttk.Combobox(mission_frame, textvariable=self.mission_var, 
                                   values=missions, state="readonly")
        mission_combo.pack(fill=tk.X, pady=5)
        mission_combo.bind('<<ComboboxSelected>>', self.on_mission_select)
        
        # Datos en tiempo real
        realtime_frame = ttk.LabelFrame(parent, text="Live NASA Data", padding=10)
        realtime_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(realtime_frame, text="🌍 Fetch Earth Image", 
                  command=self.fetch_earth_image).pack(fill=tk.X, pady=2)
        ttk.Button(realtime_frame, text="🪐 Fetch Mars Image", 
                  command=self.fetch_mars_image).pack(fill=tk.X, pady=2)
        
        # Búsqueda por coordenadas
        coords_frame = ttk.LabelFrame(parent, text="Coordinate Search", padding=10)
        coords_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(coords_frame, text="Lat/Lon:").pack(anchor=tk.W)
        self.coord_entry = ttk.Entry(coords_frame)
        self.coord_entry.pack(fill=tk.X, pady=5)
        self.coord_entry.insert(0, "40.7128, -74.0060")
        
        ttk.Button(coords_frame, text="Search", command=self.search_coordinates).pack(fill=tk.X)
        
        # Herramientas de análisis
        analysis_frame = ttk.LabelFrame(parent, text="Analysis Tools", padding=10)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(analysis_frame, text="🧠 AI Image Analysis", 
                  command=self.analizar_imagen_ia).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="🕐 Compare Images", 
                  command=self.compare_temporal_images).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="🏛️ Kiosk Mode", 
                  command=self.enable_kiosk_mode).pack(fill=tk.X, pady=2)
        
        # Filtros rápidos
        filter_frame = ttk.LabelFrame(parent, text="Quick Filters", padding=10)
        filter_frame.pack(fill=tk.X)
        
        ttk.Label(filter_frame, text="Quick Filter:").pack(anchor=tk.W)
        self.quick_filter_var = tk.StringVar()
        quick_filters = ["Original", "Grayscale", "Sepia", "Negative", "Edge Detection"]
        quick_combo = ttk.Combobox(filter_frame, textvariable=self.quick_filter_var, 
                                  values=quick_filters, state="readonly")
        quick_combo.set("Original")
        quick_combo.pack(fill=tk.X, pady=5)
        quick_combo.bind('<<ComboboxSelected>>', self.apply_quick_filter)
        
        ttk.Button(filter_frame, text="🔄 Reset to Original", 
                  command=self.reset_to_original).pack(fill=tk.X, pady=2)

    def setup_center_panel(self, parent):
        # Barra de herramientas
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        controls = ["Load Image", "Zoom In", "Zoom Out", "Fit to Window", "Label Mode", "Save Image"]
        commands = [self.load_image, self.zoom_in, self.zoom_out, self.zoom_to_fit, 
                   self.toggle_label_mode, self.save_current_image]
        
        for text, cmd in zip(controls, commands):
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=2)
        
        # Control de zoom
        zoom_frame = ttk.Frame(toolbar)
        zoom_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=1, to=1000, orient=tk.HORIZONTAL, 
                                   command=self.on_zoom_slider, length=150)
        self.zoom_slider.set(100)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        # Canvas para imagen
        canvas_container = ttk.Frame(parent)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas = tk.Canvas(canvas_container, 
                               bg="black",
                               yscrollcommand=self.v_scrollbar.set,
                               xscrollcommand=self.h_scrollbar.set,
                               cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)
        
        self.bind_events()

    def setup_bottom_panel(self, parent):
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Notebook para diferentes análisis
        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_basico = ttk.Frame(self.notebook)
        self.tab_planetario = ttk.Frame(self.notebook)
        self.tab_estelar = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_basico, text="🧠 Basic Analysis")
        self.notebook.add(self.tab_planetario, text="🪐 Planetary")
        self.notebook.add(self.tab_estelar, text="🌌 Stellar")
        
        # Áreas de texto para análisis
        self.text_basico = scrolledtext.ScrolledText(self.tab_basico, wrap=tk.WORD, 
                                                   font=('Consolas', 9), height=8)
        self.text_basico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_planetario = scrolledtext.ScrolledText(self.tab_planetario, wrap=tk.WORD,
                                                       font=('Consolas', 9), height=8)
        self.text_planetario.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_estelar = scrolledtext.ScrolledText(self.tab_estelar, wrap=tk.WORD,
                                                     font=('Consolas', 9), height=8)
        self.text_estelar.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Ready - Load a NASA dataset to begin exploration")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5, 0))

    def bind_events(self):
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        
        # Atajos de teclado
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Control-s>", lambda e: self.save_current_image())
        self.root.bind("<Control-l>", lambda e: self.toggle_label_mode())
        self.root.bind("<Control-r>", lambda e: self.reset_to_original())
        self.root.bind("<Escape>", lambda e: self.cancel_current_operation())

    # =========================================================================
    # MÉTODOS PRINCIPALES
    # =========================================================================

    def load_sample_datasets(self):
        self.sample_datasets = {
            "TESS FFI": {"description": "TESS Full Frame Image - Sector 45", "size": "4096x4096"},
            "Lunar Reconnaissance": {"description": "LRO LROC NAC Mosaic", "size": "8192x8192"}, 
            "Earth Observatory": {"description": "Landsat 8 Composite", "size": "10240x10240"}
        }

    def on_mission_select(self, event):
        mission = self.mission_var.get()
        if mission in self.nasa_datasets:
            dataset = self.nasa_datasets[mission]
            self.status_var.set(f"Selected: {mission} - {dataset['description']}")
            self.load_simulated_nasa_dataset(mission)
        elif mission in self.sample_datasets:
            dataset = self.sample_datasets[mission]
            self.status_var.set(f"Selected: {mission} - {dataset['description']}")
            self.load_simulated_image(mission)

    def load_simulated_nasa_dataset(self, dataset_name):
        if dataset_name in self.nasa_datasets:
            dataset = self.nasa_datasets[dataset_name]
            
            if dataset_name == "Hubble_Andromeda":
                image_array = self.gigapixel_handler.create_simulated_gigapixel(5000, 2500)
            elif dataset_name == "MRO_Mars_Global":
                image_array = self.create_simulated_mars_surface(4000, 2000)
            else:  # LRO_Lunar_Mosaic
                image_array = self.create_simulated_lunar_surface(6000, 3000)
            
            image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            self.set_image(image)
            self.mostrar_estado(f"✅ Loaded {dataset_name}")

    def create_simulated_mars_surface(self, width, height):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = (120, 60, 20)  # Color marciano
        
        np.random.seed(42)
        for _ in range(100):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(10, 50)
            cv2.circle(image, (x, y), radius, (100, 50, 15), -1)
            cv2.circle(image, (x, y), radius, (140, 70, 25), 2)
        
        return image

    def create_simulated_lunar_surface(self, width, height):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = (100, 100, 100)  # Color lunar
        
        np.random.seed(42)
        for _ in range(200):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(5, 30)
            cv2.circle(image, (x, y), radius, (80, 80, 80), -1)
            cv2.circle(image, (x, y), radius, (150, 150, 150), 1)
        
        return image

    def load_simulated_image(self, mission):
        width, height = 800, 800
        
        if mission == "TESS FFI":
            image = self.create_simulated_starfield(width, height)
        elif mission == "Lunar Reconnaissance":
            image = self.create_simulated_lunar_surface(width, height)
        elif mission == "Earth Observatory":
            image = self.create_simulated_earth_view(width, height)
        else:
            image = self.create_scientific_image(width, height)
            
        self.set_image(image)

    def create_simulated_starfield(self, width, height):
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)
        
        np.random.seed(42)
        for _ in range(100):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.exponential(1) + 0.5
            color = (255, 255, 255)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            
        return image

    def create_simulated_earth_view(self, width, height):
        image = Image.new('RGB', (width, height), color=(0, 0, 50))
        draw = ImageDraw.Draw(image)
        draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], 
                    fill=(0, 100, 0), outline=(0, 80, 0))
        return image

    def create_scientific_image(self, width, height):
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)
        
        for i in range(0, width, 50):
            for j in range(0, height, 50):
                intensity = (i + j) % 255
                color = (intensity, intensity//2, 255-intensity)
                draw.rectangle([i, j, i+40, j+40], fill=color)
                
        return image

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp")]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                self.set_image(image)
                self.mostrar_estado(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def set_image(self, image):
        self.original_image = image
        self.original_size = image.size
        self.scale = 1.0
        self.zoom_slider.set(100)
        
        self.cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.current_filtered_image = self.cv_image.copy()
        
        self.generar_todos_filtros_automatico()
        self.display_current_image()
        self.center_image()
        self.clear_all_filters()

    def display_current_image(self):
        if self.current_filtered_image is None:
            return
            
        if len(self.current_filtered_image.shape) == 3:
            if self.current_filtered_image.shape[2] == 3:
                image_rgb = cv2.cvtColor(self.current_filtered_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(self.current_filtered_image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(self.current_filtered_image, cv2.COLOR_GRAY2RGB)
            
        image_pil = Image.fromarray(image_rgb)
        
        width = int(image_pil.width * self.scale)
        height = int(image_pil.height * self.scale)
        resized_image = image_pil.resize((width, height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(resized_image)
        
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, width, height))
        
        self.zoom_label.config(text=f"{int(self.scale * 100)}%")
        self.mostrar_estado(f"Displaying: {width}x{height} pixels | Zoom: {int(self.scale * 100)}%")

    def center_image(self):
        if self.canvas_image:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_width = self.original_size[0] * self.scale
            img_height = self.original_size[1] * self.scale
            
            if img_width < canvas_width and img_height < canvas_height:
                x_offset = (canvas_width - img_width) / 2
                y_offset = (canvas_height - img_height) / 2
                self.canvas.coords(self.canvas_image, x_offset, y_offset)

    def aplicar_filtros_color_avanzados(self, imagen):
        filtros = {}
        filtros["Original"] = imagen
        
        # Escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        filtros["Grayscale"] = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        
        # Sepia
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia = cv2.transform(imagen, sepia_filter)
        filtros["Sepia"] = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Negativo
        filtros["Negative"] = 255 - imagen
        
        # Detección de bordes
        bordes = cv2.Canny(gris, 100, 200)
        filtros["Edge Detection"] = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
        
        # Desenfoque gaussiano
        filtros["Gaussian Blur"] = cv2.GaussianBlur(imagen, (15, 15), 0)
        
        return filtros

    def generar_todos_filtros_automatico(self):
        if self.cv_image is None:
            return
            
        self.mostrar_estado("⚡ Generating filters...")
        
        def generar_thread():
            try:
                todos_filtros = self.aplicar_filtros_color_avanzados(self.cv_image)
                for nombre, filtro in todos_filtros.items():
                    self.transformaciones[nombre] = filtro
                self.root.after(0, lambda: self.mostrar_estado("✅ Filters generated"))
            except Exception as e:
                self.root.after(0, lambda: self.mostrar_estado("❌ Error generating filters"))
        
        threading.Thread(target=generar_thread, daemon=True).start()

    def apply_quick_filter(self, event=None):
        filter_name = self.quick_filter_var.get()
        if filter_name in self.transformaciones:
            self.apply_filter_to_main(filter_name)

    def apply_filter_to_main(self, filter_name):
        if filter_name in self.transformaciones:
            self.current_filtered_image = self.transformaciones[filter_name]
            self.display_current_image()
            self.mostrar_estado(f"✅ Filter applied: {filter_name}")

    def reset_to_original(self):
        self.current_filtered_image = self.cv_image.copy()
        self.display_current_image()
        self.mostrar_estado("🔄 Image restored to original")

    def save_current_image(self):
        if self.current_filtered_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
        )
        
        if ruta:
            try:
                cv2.imwrite(ruta, self.current_filtered_image)
                messagebox.showinfo("Success", f"Image saved as: {ruta}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {e}")

    # =========================================================================
    # MÉTODOS NASA APIs
    # =========================================================================

    def fetch_earth_image(self):
        try:
            coords = self.coord_entry.get().split(',')
            if len(coords) == 2:
                lat, lon = float(coords[0].strip()), float(coords[1].strip())
            else:
                lat, lon = 40.7128, -74.0060
            
            self.mostrar_estado("🌍 Fetching Earth image from NASA...")
            
            def fetch_thread():
                image = self.nasa_fetcher.fetch_earth_imagery(lat, lon)
                if image:
                    self.root.after(0, lambda: self.set_image(image))
                    self.root.after(0, lambda: self.mostrar_estado("✅ Earth image loaded"))
                else:
                    self.root.after(0, lambda: self.mostrar_estado("❌ Failed to fetch Earth image"))
            
            threading.Thread(target=fetch_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch Earth image: {e}")

    def fetch_mars_image(self):
        self.mostrar_estado("🪐 Fetching Mars image (simulated)...")
        # Por simplicidad, usamos una imagen simulada
        mars_image = self.create_simulated_mars_surface(800, 600)
        image = Image.fromarray(cv2.cvtColor(mars_image, cv2.COLOR_BGR2RGB))
        self.set_image(image)
        self.mostrar_estado("✅ Mars image loaded")

    def search_coordinates(self):
        coordinates = self.coord_entry.get()
        if coordinates:
            self.mostrar_estado(f"Searching coordinates: {coordinates}")
            messagebox.showinfo("Search", f"Centering on coordinates: {coordinates}")

    # =========================================================================
    # MÉTODOS DE ANÁLISIS
    # =========================================================================

    def analizar_imagen_ia(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.mostrar_estado("🔍 Starting AI analysis...")
        
        def analysis_thread():
            try:
                # Análisis básico
                resultado_basico = self._analisis_basico()
                self.root.after(0, lambda: self.actualizar_resultado_basico(resultado_basico))
                
                # Análisis planetario
                resultado_planetario = self._analisis_planetario()
                self.root.after(0, lambda: self.actualizar_resultado_planetario(resultado_planetario))
                
                # Análisis estelar
                resultado_estelar = self._analisis_estelar()
                self.root.after(0, lambda: self.actualizar_resultado_estelar(resultado_estelar))
                
                self.root.after(0, lambda: self.mostrar_estado("✅ AI analysis completed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.mostrar_estado(f"❌ Analysis error: {e}"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()

    def _analisis_basico(self):
        resultado = "1️⃣  BASIC ANALYSIS\n" + "="*50 + "\n\n"
        resultado += f"📊 Image Information:\n"
        resultado += f"• Resolution: {self.original_size[0]} x {self.original_size[1]} pixels\n"
        resultado += f"• Format: {self.original_image.format if hasattr(self.original_image, 'format') else 'Unknown'}\n"
        resultado += f"• Mode: {self.original_image.mode}\n\n"
        
        resultado += "🔍 NASA Context:\n"
        resultado += "• This image can be analyzed for scientific features\n"
        resultado += "• Use pattern detection for celestial objects\n"
        resultado += "• Compare with NASA databases for identification\n"
        
        return resultado

    def _analisis_planetario(self):
        resultado = "2️⃣  PLANETARY ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "🪐 Planetary Features Analysis:\n\n"
        resultado += "Potential features to analyze:\n"
        resultado += "• Craters and impact structures\n"
        resultado += "• Geological formations\n"
        resultado += "• Atmospheric patterns\n"
        resultado += "• Surface composition variations\n\n"
        
        resultado += "🔬 Recommended NASA Missions:\n"
        resultado += "• Mars Reconnaissance Orbiter (MRO)\n"
        resultado += "• Lunar Reconnaissance Orbiter (LRO)\n"
        resultado += "• Mars Rover missions\n"
        
        return resultado

    def _analisis_estelar(self):
        resultado = "3️⃣  STELLAR ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "🌌 Stellar and Galactic Analysis:\n\n"
        resultado += "Potential astronomical features:\n"
        resultado += "• Star clusters and formations\n"
        resultado += "• Galactic structures\n"
        resultado += "• Nebulae and interstellar clouds\n"
        resultado += "• Cosmic background patterns\n\n"
        
        resultado += "🔭 Recommended NASA Missions:\n"
        resultado += "• Hubble Space Telescope\n"
        resultado += "• James Webb Space Telescope\n"
        resultado += "• Chandra X-ray Observatory\n"
        
        return resultado

    def actualizar_resultado_basico(self, texto):
        self.text_basico.delete(1.0, tk.END)
        self.text_basico.insert(1.0, texto)
        self.notebook.select(0)

    def actualizar_resultado_planetario(self, texto):
        self.text_planetario.delete(1.0, tk.END)
        self.text_planetario.insert(1.0, texto)

    def actualizar_resultado_estelar(self, texto):
        self.text_estelar.delete(1.0, tk.END)
        self.text_estelar.insert(1.0, texto)

    def compare_temporal_images(self):
        if self.current_filtered_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        # Simular comparación temporal
        image1 = self.cv_image
        image2 = cv2.GaussianBlur(self.cv_image, (25, 25), 0)
        
        result = self.temporal_analyzer.compare_temporal_images(image1, image2)
        
        report = f"TEMPORAL ANALYSIS REPORT\n{'='*40}\n\n"
        report += f"Changes detected: {result['changes_detected']}\n"
        report += f"Percent change: {result['percent_change']:.2f}%\n"
        
        self.text_basico.delete(1.0, tk.END)
        self.text_basico.insert(1.0, report)
        self.notebook.select(0)
        
        self.mostrar_estado(f"🕐 Temporal analysis: {result['changes_detected']} changes detected")

    # =========================================================================
    # MÉTODOS DE NAVEGACIÓN Y ZOOM
    # =========================================================================

    def zoom_in(self, event=None, factor=1.2):
        if self.current_filtered_image is None: return
        self.scale *= factor
        self.scale = max(0.01, min(50.0, self.scale))
        self.display_current_image()
        self.zoom_slider.set(self.scale * 100)
        
    def zoom_out(self, event=None, factor=1.2):
        self.zoom_in(factor=1/factor)
        
    def on_mousewheel(self, event):
        if self.current_filtered_image is None: return
        if event.delta > 0 or event.num == 4:
            self.zoom_in()
        else:
            self.zoom_out()
            
    def zoom_to_fit(self):
        if self.current_filtered_image is None: return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / self.original_size[0]
            scale_y = canvas_height / self.original_size[1]
            self.scale = min(scale_x, scale_y, 1.0)
            self.zoom_slider.set(self.scale * 100)
            self.display_current_image()
            self.center_image()

    def on_zoom_slider(self, value):
        if self.current_filtered_image is not None:
            self.scale = float(value) / 100.0
            self.display_current_image()

    # =========================================================================
    # MÉTODOS DE ANOTACIONES
    # =========================================================================

    def toggle_label_mode(self):
        self.label_mode = not self.label_mode
        if self.label_mode:
            self.canvas.config(cursor="crosshair")
            self.mostrar_estado("Label Mode: Click and drag to mark features")
        else:
            self.canvas.config(cursor="crosshair")
            self.mostrar_estado("Navigation Mode")

    def on_button_press(self, event):
        if self.label_mode and self.current_filtered_image is not None:
            self.start_x = self.canvas.canvasx(event.x)
            self.start_y = self.canvas.canvasy(event.y)
            self.current_label = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline="red", width=2, dash=(4, 2), tags="temp_label"
            )
        else:
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.is_dragging = True

    def on_mouse_drag(self, event):
        if self.current_label:
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.current_label, self.start_x, self.start_y, end_x, end_y)
        elif self.is_dragging:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.canvas.xview_scroll(-dx, "units")
            self.canvas.yview_scroll(-dy, "units")
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def on_button_release(self, event):
        if self.current_label:
            self.finalize_label(event)
        elif self.is_dragging:
            self.is_dragging = False

    def on_mouse_move(self, event):
        if self.current_filtered_image is not None:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            if 0 <= x < self.original_size[0] * self.scale and 0 <= y < self.original_size[1] * self.scale:
                orig_x = int(x / self.scale)
                orig_y = int(y / self.scale)
                self.mostrar_estado(f"Position: ({orig_x}, {orig_y}) | Zoom: {int(self.scale * 100)}%")

    def on_double_click(self, event):
        if self.current_filtered_image is None: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = self.original_size[0] * self.scale
        img_height = self.original_size[1] * self.scale
        if img_width > canvas_width:
            x_frac = max(0, min(1, (x - canvas_width/2) / img_width))
            self.canvas.xview_moveto(x_frac)
        if img_height > canvas_height:
            y_frac = max(0, min(1, (y - canvas_height/2) / img_height))
            self.canvas.yview_moveto(y_frac)

    def finalize_label(self, event):
        if self.current_label:
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)
            orig_start_x = int(self.start_x / self.scale)
            orig_start_y = int(self.start_y / self.scale)
            orig_end_x = int(end_x / self.scale)
            orig_end_y = int(end_y / self.scale)
            
            if abs(orig_end_x - orig_start_x) > 5 and abs(orig_end_y - orig_start_y) > 5:
                label_name = simpledialog.askstring("Feature Label", "Enter feature name or description:")
                if label_name:
                    feature_type = simpledialog.askstring("Feature Type", 
                        "Enter feature type (crater, star, galaxy, etc):", initialvalue="unknown")
                    
                    ann_id = self.collab_annotations.add_annotation(
                        "current", orig_start_x, orig_start_y, orig_end_x, orig_end_y,
                        label_name, "user", feature_type or "unknown"
                    )
                    
                    self.canvas.itemconfig(self.current_label, dash=(), fill="", 
                                         outline="yellow", width=2, tags="permanent_label")
                    text_id = self.canvas.create_text(
                        self.start_x, self.start_y - 10,
                        text=label_name, fill="yellow", anchor=tk.SW, tags="permanent_label"
                    )
                    
                    self.mostrar_estado(f"Label added: {label_name} ({feature_type})")
            else:
                self.canvas.delete(self.current_label)
            self.current_label = None

    def cancel_current_operation(self):
        if self.current_label:
            self.canvas.delete(self.current_label)
            self.current_label = None
        self.label_mode = False
        self.mostrar_estado("Operation cancelled")

    def enable_kiosk_mode(self):
        self.museum_mode.enable_kiosk_mode()
        self.mostrar_estado("🏛️ Kiosk mode enabled")

    def clear_all_filters(self):
        self.active_filters.clear()
        self.mostrar_estado("🔄 All filters cleared")

    def mostrar_estado(self, mensaje):
        self.status_var.set(mensaje)

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    try:
        root = tk.Tk()
        app = NASAImageExplorerPro(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()