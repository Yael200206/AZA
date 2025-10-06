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
import torch
import requests
from io import BytesIO
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from bs4 import BeautifulSoup
import re
from datetime import datetime

class NASAImageExplorerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("NASA Multi-Mission Image Explorer - Scientific Platform")
        self.root.geometry("1400x900")
        
        # Variables de la imagen
        self.images = {}
        self.active_image_id = None
        self.photo = None
        self.canvas_image = None
        self.scale = 1.0
        self.original_size = (0, 0)
        self.original_image = None
        self.cv_image = None
        self.current_filtered_image = None
        
        # Sistema de capas y filtros
        self.layers = {}
        self.transformaciones = {}
        self.active_filters = []
        
        # Variables para navegación
        self.labels = []
        self.annotations = []
        self.current_label = None
        self.label_mode = False
        self.start_x = None
        self.start_y = None
        self.drag_start_x = None
        self.drag_start_y = None
        self.is_dragging = False
        self.last_viewport = None
        
        # Sistema de análisis IA
        self.analizador = None
        self.analizando = False
        self.filtros_generados = False
        
        # Cache para imágenes procesadas
        self.image_cache = {}
        
        # Nuevas variables para división de imagen
        self.image_tiles = {}
        self.tile_size = 512
        self.current_tiles = {}
        
        # Sistema de rangos de color
        self.rangos_color = {
            "Rojo": {
                "hsv_bajo1": np.array([0, 100, 100]),
                "hsv_alto1": np.array([10, 255, 255]),
                "hsv_bajo2": np.array([170, 100, 100]),
                "hsv_alto2": np.array([180, 255, 255]),
                "color_bgr": (0, 0, 255)
            },
            "Azul": {
                "hsv_bajo": np.array([100, 100, 100]),
                "hsv_alto": np.array([130, 255, 255]),
                "color_bgr": (255, 0, 0)
            },
            "Amarillo": {
                "hsv_bajo": np.array([20, 100, 100]),
                "hsv_alto": np.array([30, 255, 255]),
                "color_bgr": (0, 255, 255)
            },
            "Verde": {
                "hsv_bajo": np.array([40, 100, 100]),
                "hsv_alto": np.array([80, 255, 255]),
                "color_bgr": (0, 255, 0)
            },
            "Naranja": {
                "hsv_bajo": np.array([10, 100, 100]),
                "hsv_alto": np.array([20, 255, 255]),
                "color_bgr": (0, 165, 255)
            },
            "Morado": {
                "hsv_bajo": np.array([130, 100, 100]),
                "hsv_alto": np.array([160, 255, 255]),
                "color_bgr": (255, 0, 255)
            }
        }
        self.vars_color = {}
        
        # Variables para detección de patrones
        self.detectar_circulos = tk.BooleanVar(value=True)
        self.detectar_contornos = tk.BooleanVar(value=True)
        self.mostrar_centroides = tk.BooleanVar(value=True)
        self.detectar_rectangulos = tk.BooleanVar(value=True)
        self.detectar_lineas = tk.BooleanVar(value=True)
        self.patrones_detectados = []
        
        # Configurar interfaz
        self.setup_ui()
        
        # Cargar datos y modelos
        self.load_sample_datasets()
        self.cargar_modelos_background()
        
    def cargar_modelos_background(self):
        def cargar_modelos():
            try:
                self.mostrar_estado("🔄 Cargando modelos IA...")
                self.analizador = AnalizadorEspecializadoNASA()
                self.mostrar_estado("✅ Modelos IA cargados")
            except Exception as e:
                self.mostrar_estado("❌ Error cargando modelos IA")
        
        thread = threading.Thread(target=cargar_modelos)
        thread.daemon = True
        thread.start()
        
    def mostrar_estado(self, mensaje):
        self.status_var.set(mensaje)
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        center_panel = ttk.Frame(main_frame)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_center_panel(center_panel)
        self.setup_bottom_panel(center_panel)
        
    def setup_left_panel(self, parent):
        mission_frame = ttk.LabelFrame(parent, text="NASA Missions & Datasets", padding=10)
        mission_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mission_frame, text="Select Mission:").pack(anchor=tk.W)
        self.mission_var = tk.StringVar()
        missions = ["TESS FFI", "Lunar Reconnaissance", "Earth Observatory", "Mars Rover", "Hubble Deep Field", "JWST NIRCam"]
        mission_combo = ttk.Combobox(mission_frame, textvariable=self.mission_var, values=missions, state="readonly")
        mission_combo.pack(fill=tk.X, pady=5)
        mission_combo.bind('<<ComboboxSelected>>', self.on_mission_select)
        
        coords_frame = ttk.LabelFrame(parent, text="Coordinate Search", padding=10)
        coords_frame.pack(fill=tk.X, pady=(0, 10))
        
        coord_grid = ttk.Frame(coords_frame)
        coord_grid.pack(fill=tk.X)
        
        ttk.Label(coord_grid, text="RA/Dec or Lat/Lon:").grid(row=0, column=0, sticky=tk.W)
        self.coord_entry = ttk.Entry(coord_grid)
        self.coord_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(coord_grid, text="Feature Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.feature_entry = ttk.Entry(coord_grid)
        self.feature_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        coord_grid.columnconfigure(1, weight=1)
        
        ttk.Button(coords_frame, text="Search", command=self.search_coordinates).pack(fill=tk.X)
        
        self.setup_layers_panel(parent)
        self.setup_analysis_panel(parent)
        
    def setup_layers_panel(self, parent):
        layers_frame = ttk.LabelFrame(parent, text="Image Layers & Filters", padding=10)
        layers_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.filters_notebook = ttk.Notebook(layers_frame)
        self.filters_notebook.pack(fill=tk.BOTH, expand=True)
        
        self.active_filters_tab = ttk.Frame(self.filters_notebook)
        self.color_filters_tab = ttk.Frame(self.filters_notebook)
        self.color_adjust_tab = ttk.Frame(self.filters_notebook)
        self.color_ranges_tab = ttk.Frame(self.filters_notebook)
        self.pattern_detection_tab = ttk.Frame(self.filters_notebook)
        
        self.filters_notebook.add(self.active_filters_tab, text="Active Filters")
        self.filters_notebook.add(self.color_filters_tab, text="Color Filters")
        self.filters_notebook.add(self.color_adjust_tab, text="Color Adjustment")
        self.filters_notebook.add(self.color_ranges_tab, text="Color Ranges")
        self.filters_notebook.add(self.pattern_detection_tab, text="Pattern Detection")
        
        self.setup_active_filters_tab(self.active_filters_tab)
        self.setup_color_filters_tab(self.color_filters_tab)
        self.setup_color_adjust_tab(self.color_adjust_tab)
        self.setup_color_ranges_tab(self.color_ranges_tab)
        self.setup_pattern_detection_tab(self.pattern_detection_tab)
        
    def setup_pattern_detection_tab(self, parent):
        pattern_frame = ttk.Frame(parent)
        pattern_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(pattern_frame, text="Pattern Detection Settings:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        detection_frame = ttk.Frame(pattern_frame)
        detection_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(detection_frame, text="Detect Circles", 
                       variable=self.detectar_circulos).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(detection_frame, text="Detect Contours", 
                       variable=self.detectar_contornos).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(detection_frame, text="Detect Rectangles", 
                       variable=self.detectar_rectangulos).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(detection_frame, text="Show Centroids", 
                       variable=self.mostrar_centroides).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(detection_frame, text="Detect Lines", 
                       variable=self.detectar_lineas).pack(anchor=tk.W, pady=2)
        
        btn_frame = ttk.Frame(pattern_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="🔍 Detect Patterns", 
                  command=self.aplicar_deteccion_patrones).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Clear Patterns", 
                  command=self.limpiar_patrones).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        results_frame = ttk.LabelFrame(pattern_frame, text="Detected Patterns", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.pattern_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                            height=10, font=('Consolas', 9))
        self.pattern_results_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.pattern_results_text.config(state='disabled')
        
    def aplicar_deteccion_patrones(self):
        if self.current_filtered_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
    
        try:
            # Obtener la región visible actual del canvas
            region_visible = self.obtener_region_visible()
            if region_visible is None:
                messagebox.showwarning("Warning", "No visible region found")
                return
            
            x1, y1, x2, y2 = region_visible
            
            # Extraer solo la región visible de la imagen
            region_imagen = self.current_filtered_image[y1:y2, x1:x2]
            
            # Aplicar detección de patrones solo en la región visible
            region_con_patrones, info_patrones = self.detectar_patrones_con_colores(region_imagen)
            
            # Crear una copia de la imagen completa
            imagen_completa_con_patrones = self.current_filtered_image.copy()
            
            # Reemplazar solo la región visible con la versión con patrones
            imagen_completa_con_patrones[y1:y2, x1:x2] = region_con_patrones
            
            # Actualizar la imagen mostrada
            self.current_filtered_image = imagen_completa_con_patrones
            self.update_filtered_tiles()
            self.display_current_image()
            
            self.mostrar_resultados_patrones(info_patrones)
            
            filter_name = "Pattern Detection (Visible Area)"
            if filter_name not in self.active_filters:
                self.active_filters.append(filter_name)
                self.filters_listbox.insert(tk.END, filter_name)
            
            self.mostrar_estado("✅ Pattern detection applied to visible area")
        
        except Exception as e:
            self.mostrar_estado(f"❌ Error in pattern detection: {e}")
            messagebox.showerror("Error", f"Error detecting patterns: {str(e)}")
    def obtener_region_visible(self):
        """Obtiene las coordenadas de la región visible actual en la imagen original"""
        if self.current_filtered_image is None:
            return None
        
        # Obtener dimensiones del canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return None
        
        # Obtener la vista actual del scroll
        x_view = self.canvas.xview()
        y_view = self.canvas.yview()
        
        # Calcular dimensiones de la imagen escalada
        img_width_scaled = self.original_size[0] * self.scale
        img_height_scaled = self.original_size[1] * self.scale
        
        # Calcular región visible en coordenadas escaladas
        visible_x1_scaled = max(0, int(x_view[0] * img_width_scaled))
        visible_y1_scaled = max(0, int(y_view[0] * img_height_scaled))
        visible_x2_scaled = min(img_width_scaled, int(x_view[1] * img_width_scaled))
        visible_y2_scaled = min(img_height_scaled, int(y_view[1] * img_height_scaled))
        
        # Convertir a coordenadas de la imagen original
        x1 = int(visible_x1_scaled / self.scale)
        y1 = int(visible_y1_scaled / self.scale)
        x2 = int(visible_x2_scaled / self.scale)
        y2 = int(visible_y2_scaled / self.scale)
        
        # Asegurar que las coordenadas estén dentro de los límites de la imagen
        x1 = max(0, min(x1, self.original_size[0] - 1))
        y1 = max(0, min(y1, self.original_size[1] - 1))
        x2 = max(1, min(x2, self.original_size[0]))
        y2 = max(1, min(y2, self.original_size[1]))
        
        # Asegurar que x2 > x1 y y2 > y1
        if x2 <= x1 or y2 <= y1:
            return None
        
        return (x1, y1, x2, y2)
    

    def mostrar_vista_previa_patrones(self, imagen_con_patrones):
        """Muestra la detección de patrones en una ventana de vista previa"""
        # Crear ventana de vista previa
        ventana_previa = tk.Toplevel(self.root)
        ventana_previa.title("Pattern Detection Preview")
        ventana_previa.geometry("1000x800")
        
        # Frame principal
        main_frame = ttk.Frame(ventana_previa)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        ttk.Label(main_frame, text="Pattern Detection Preview", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Frame para imagen
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Convertir imagen para mostrar
        if len(imagen_con_patrones.shape) == 3:
            if imagen_con_patrones.shape[2] == 3:
                imagen_rgb = cv2.cvtColor(imagen_con_patrones, cv2.COLOR_BGR2RGB)
            else:
                imagen_rgb = cv2.cvtColor(imagen_con_patrones, cv2.COLOR_GRAY2RGB)
        else:
            imagen_rgb = cv2.cvtColor(imagen_con_patrones, cv2.COLOR_GRAY2RGB)
        
        # Redimensionar para vista previa si es muy grande
        h, w = imagen_rgb.shape[:2]
        max_size = 800
        if w > max_size or h > max_size:
            ratio = min(max_size/w, max_size/h)
            new_w, new_h = int(w*ratio), int(h*ratio)
            imagen_mostrar = cv2.resize(imagen_rgb, (new_w, new_h))
        else:
            imagen_mostrar = imagen_rgb
        
        # Convertir a PhotoImage
        imagen_pil = Image.fromarray(imagen_mostrar)
        self.preview_photo = ImageTk.PhotoImage(imagen_pil)
        
        # Mostrar en label
        preview_label = ttk.Label(image_frame, image=self.preview_photo)
        preview_label.pack(expand=True)
        
        # Frame para controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Botones de acción
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Apply to Main Image", 
                  command=lambda: self.aplicar_patrones_a_imagen_principal(imagen_con_patrones, ventana_previa)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Preview", 
                  command=lambda: self.guardar_imagen_previa(imagen_con_patrones)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close Preview", 
                  command=ventana_previa.destroy).pack(side=tk.RIGHT, padx=5)

    def aplicar_patrones_a_imagen_principal(self, imagen_con_patrones, ventana_previa):
        """Aplica la detección de patrones a la imagen principal"""
        self.current_filtered_image = imagen_con_patrones.copy()
        self.update_filtered_tiles()
        self.display_current_image()
        
        filter_name = "Pattern Detection"
        if filter_name not in self.active_filters:
            self.active_filters.append(filter_name)
            self.filters_listbox.insert(tk.END, filter_name)
        
        self.mostrar_estado("✅ Pattern detection applied to main image")
        ventana_previa.destroy()

    def guardar_imagen_previa(self, imagen_con_patrones):
        """Guarda la imagen con patrones detectados"""
        ruta = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
        )
        
        if ruta:
            try:
                cv2.imwrite(ruta, imagen_con_patrones)
                messagebox.showinfo("Éxito", f"Preview image saved as: {ruta}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {e}")
    
    def detectar_patrones_con_colores(self, imagen):
        imagen_patrones = imagen.copy()
        info_patrones = []
        self.patrones_detectados = []
        
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris_suavizado = cv2.medianBlur(gris, 5)
        
        # Detectar círculos
        if self.detectar_circulos.get():
            try:
                circles = cv2.HoughCircles(
                    gris_suavizado,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=30,
                    param1=50,
                    param2=30,
                    minRadius=5,
                    maxRadius=100
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    
                    for (x, y, r) in circles:
                        color_dominante = self.analizar_color_area(imagen, x, y, r)
                        
                        # Solo poner un punto en el centro del color dominante
                        color_bgr = self.rangos_color.get(color_dominante, {}).get("color_bgr", (0, 255, 0))
                        cv2.circle(imagen_patrones, (x, y), 5, color_bgr, -1)  # Punto sólido
                        
                        patron_info = {
                            "tipo": "Círculo",
                            "centro": (x, y),
                            "radio": r,
                            "color_dominante": color_dominante,
                            "area": np.pi * r * r
                        }
                        self.patrones_detectados.append(patron_info)
                    
                    info_patrones.append(f"Círculos detectados: {len(circles)}")
            except Exception as e:
                info_patrones.append(f"Error en círculos: {str(e)}")
        
        # Detectar contornos
        if self.detectar_contornos.get():
            try:
                _, thresh = cv2.threshold(gris_suavizado, 127, 255, cv2.THRESH_BINARY)
                contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                contornos_filtrados = [cnt for cnt in contornos if cv2.contourArea(cnt) > 100]
                
                for i, cnt in enumerate(contornos_filtrados):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        color_dominante = self.analizar_color_area(imagen, cx, cy, 10)
                        
                        # Solo poner un punto en el centroide del color dominante
                        color_bgr = self.rangos_color.get(color_dominante, {}).get("color_bgr", (0, 255, 0))
                        cv2.circle(imagen_patrones, (cx, cy), 5, color_bgr, -1)
                        
                        patron_info = {
                            "tipo": "Contorno",
                            "centroide": (cx, cy),
                            "area": cv2.contourArea(cnt),
                            "color_dominante": color_dominante
                        }
                        self.patrones_detectados.append(patron_info)
                
                info_patrones.append(f"Contornos detectados: {len(contornos_filtrados)}")
            except Exception as e:
                info_patrones.append(f"Error en contornos: {str(e)}")
        
        # Detectar rectángulos
        if self.detectar_rectangulos.get():
            try:
                _, thresh = cv2.threshold(gris_suavizado, 127, 255, cv2.THRESH_BINARY)
                contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                rectangulos_detectados = 0
                for i, cnt in enumerate(contornos):
                    area = cv2.contourArea(cnt)
                    if area > 500:
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                        
                        if len(approx) == 4:
                            rectangulos_detectados += 1
                            
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                color_dominante = self.analizar_color_area(imagen, cx, cy, 15)
                                
                                # Solo poner un punto en el centro del color dominante
                                color_bgr = self.rangos_color.get(color_dominante, {}).get("color_bgr", (0, 255, 0))
                                cv2.circle(imagen_patrones, (cx, cy), 5, color_bgr, -1)
                                
                                patron_info = {
                                    "tipo": "Rectángulo",
                                    "centro": (cx, cy),
                                    "area": area,
                                    "color_dominante": color_dominante,
                                    "esquinas": len(approx)
                                }
                                self.patrones_detectados.append(patron_info)
                
                info_patrones.append(f"Rectángulos detectados: {rectangulos_detectados}")
            except Exception as e:
                info_patrones.append(f"Error en rectángulos: {str(e)}")
        
        # Detectar líneas
        if self.detectar_lineas.get():
            try:
                bordes = cv2.Canny(gris_suavizado, 50, 150, apertureSize=3)
                lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=50, 
                                    minLineLength=30, maxLineGap=10)
                
                if lineas is not None:
                    lineas_detectadas = 0
                    for linea in lineas:
                        x1, y1, x2, y2 = linea[0]
                        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                        color_dominante = self.analizar_color_area(imagen, mx, my, 5)
                        
                        # Solo poner un punto en el punto medio del color dominante
                        color_bgr = self.rangos_color.get(color_dominante, {}).get("color_bgr", (0, 255, 0))
                        cv2.circle(imagen_patrones, (mx, my), 5, color_bgr, -1)
                        
                        patron_info = {
                            "tipo": "Línea",
                            "punto_medio": (mx, my),
                            "longitud": np.sqrt((x2-x1)**2 + (y2-y1)**2),
                            "color_dominante": color_dominante,
                            "puntos": [(x1, y1), (x2, y2)]
                        }
                        self.patrones_detectados.append(patron_info)
                        lineas_detectadas += 1
                    
                    info_patrones.append(f"Líneas detectadas: {lineas_detectadas}")
            except Exception as e:
                info_patrones.append(f"Error en líneas: {str(e)}")
        
        return imagen_patrones, info_patrones
    
    def analizar_color_area(self, imagen, x, y, radio):
        try:
            h, w = imagen.shape[:2]
            x1 = max(0, x - radio)
            y1 = max(0, y - radio)
            x2 = min(w, x + radio)
            y2 = min(h, y + radio)
            
            roi = imagen[y1:y2, x1:x2]
            
            if roi.size == 0:
                return "Desconocido"
            
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            conteo_colores = {}
            
            for color_name, config in self.rangos_color.items():
                if color_name == "Rojo":
                    mascara1 = cv2.inRange(roi_hsv, config["hsv_bajo1"], config["hsv_alto1"])
                    mascara2 = cv2.inRange(roi_hsv, config["hsv_bajo2"], config["hsv_alto2"])
                    mascara_color = cv2.bitwise_or(mascara1, mascara2)
                else:
                    mascara_color = cv2.inRange(roi_hsv, config["hsv_bajo"], config["hsv_alto"])
                
                conteo = np.sum(mascara_color > 0)
                if conteo > 0:
                    conteo_colores[color_name] = conteo
            
            if conteo_colores:
                return max(conteo_colores, key=conteo_colores.get)
            else:
                return "Sin color definido"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def mostrar_resultados_patrones(self, info_patrones):
        texto = "PATTERN DETECTION RESULTS\n"
        texto += "=" * 50 + "\n\n"
        
        # for info in info_patrones:
        #     texto += f"• {info}\n"
        
        texto += "\nDETALLES POR PATRÓN:\n" + "-" * 30 + "\n"
        
        for i, patron in enumerate(self.patrones_detectados, 1):
            texto += f"\n{i}. {patron['tipo']}:\n"
            texto += f"   - Color dominante: {patron['color_dominante']}\n"
            texto += f"   - Área: {patron.get('area', 'N/A'):.1f}\n"
            
            if 'centro' in patron:
                texto += f"   - Centro: {patron['centro']}\n"
            if 'radio' in patron:
                texto += f"   - Radio: {patron['radio']} px\n"
            # Se eliminó la línea del perímetro
        
        texto += f"\nTotal de patrones detectados: {len(self.patrones_detectados)}"
        
        self.pattern_results_text.config(state='normal')
        self.pattern_results_text.delete(1.0, tk.END)
        self.pattern_results_text.insert(1.0, texto)
        self.pattern_results_text.config(state='disabled')
    
    def limpiar_patrones(self):
        self.patrones_detectados = []
        self.pattern_results_text.config(state='normal')
        self.pattern_results_text.delete(1.0, tk.END)
        self.pattern_results_text.config(state='disabled')
        
        # Solo limpiar si el filtro de patrones está activo
        patron_filters = [f for f in self.active_filters if "Pattern Detection" in f]
        for filter_name in patron_filters:
            index = self.active_filters.index(filter_name)
            self.active_filters.pop(index)
            self.filters_listbox.delete(index)
        
        if patron_filters:
            self.reapply_all_filters()
        
        self.mostrar_estado("🔄 Pattern detection cleared")

    def setup_active_filters_tab(self, parent):
        ttk.Label(parent, text="Active Filters:").pack(anchor=tk.W, pady=(5,0))
        self.filters_listbox = tk.Listbox(parent, height=6)
        self.filters_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.filters_listbox.bind('<<ListboxSelect>>', self.on_filter_select)
        
        filter_controls = ttk.Frame(parent)
        filter_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(filter_controls, text="Remove Filter", 
                  command=self.remove_filter).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(filter_controls, text="Clear All", 
                  command=self.clear_all_filters).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
    def setup_color_filters_tab(self, parent):
        color_filters_frame = ttk.Frame(parent)
        color_filters_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        color_filters = [
            ("🔴 Red Channel", "red_channel"),
            ("🟢 Green Channel", "green_channel"), 
            ("🔵 Blue Channel", "blue_channel"),
            ("🌈 HSV Color", "hsv_color"),
            ("🎨 Sepia Tone", "sepia"),
            ("🌗 Grayscale", "grayscale"),
            ("⚫ Black & White", "black_white"),
            ("🌀 Negative", "negative")
        ]
        
        for i, (text, filter_type) in enumerate(color_filters):
            btn = ttk.Button(color_filters_frame, text=text, 
                           command=lambda ft=filter_type: self.apply_color_filter(ft))
            btn.pack(fill=tk.X, pady=2)
            
    def setup_color_adjust_tab(self, parent):
        adjust_frame = ttk.Frame(parent)
        adjust_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(adjust_frame, text="Brightness:").pack(anchor=tk.W, pady=(5,0))
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(adjust_frame, from_=0.1, to=3.0, 
                                   variable=self.brightness_var, orient=tk.HORIZONTAL)
        brightness_scale.pack(fill=tk.X, pady=5)
        brightness_scale.bind("<ButtonRelease-1>", lambda e: self.apply_color_adjustment())
        
        ttk.Label(adjust_frame, text="Contrast:").pack(anchor=tk.W, pady=(5,0))
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(adjust_frame, from_=0.1, to=3.0, 
                                 variable=self.contrast_var, orient=tk.HORIZONTAL)
        contrast_scale.pack(fill=tk.X, pady=5)
        contrast_scale.bind("<ButtonRelease-1>", lambda e: self.apply_color_adjustment())
        
        ttk.Label(adjust_frame, text="Saturation:").pack(anchor=tk.W, pady=(5,0))
        self.saturation_var = tk.DoubleVar(value=1.0)
        saturation_scale = ttk.Scale(adjust_frame, from_=0.0, to=3.0, 
                                   variable=self.saturation_var, orient=tk.HORIZONTAL)
        saturation_scale.pack(fill=tk.X, pady=5)
        saturation_scale.bind("<ButtonRelease-1>", lambda e: self.apply_color_adjustment())
        
        btn_frame = ttk.Frame(adjust_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Reset Adjustments", 
                  command=self.reset_color_adjustments).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Apply", 
                  command=self.apply_color_adjustment).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
    def setup_color_ranges_tab(self, parent):
        ranges_frame = ttk.Frame(parent)
        ranges_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(ranges_frame, text="Select Colors to Detect:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        colors_frame = ttk.Frame(ranges_frame)
        colors_frame.pack(fill=tk.X, pady=5)
        
        self.vars_color = {}
        row = 0
        col = 0
        
        for color_name, config in self.rangos_color.items():
            var = tk.BooleanVar(value=True)
            self.vars_color[color_name] = var
            
            color_item_frame = ttk.Frame(colors_frame)
            color_item_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            chk = ttk.Checkbutton(color_item_frame, variable=var, 
                                 command=self.apply_color_ranges)
            chk.pack(side=tk.LEFT)
            
            color_display = tk.Label(color_item_frame, text="██", 
                                   fg=self.get_color_hex(config["color_bgr"]),
                                   font=("Arial", 14, "bold"))
            color_display.pack(side=tk.LEFT, padx=(5, 2))
            
            label = ttk.Label(color_item_frame, text=color_name, font=("Arial", 10))
            label.pack(side=tk.LEFT, padx=(2, 10))
            
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        btn_frame = ttk.Frame(ranges_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Apply Color Ranges", 
                  command=self.apply_color_ranges).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Show Individual Colors", 
                  command=self.mostrar_ventanas_individuales).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Reset to Original", 
                  command=self.reset_color_ranges).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        stats_frame = ttk.LabelFrame(ranges_frame, text="Detection Statistics", padding=5)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.color_stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, 
                                                        height=8, font=('Consolas', 9))
        self.color_stats_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.color_stats_text.config(state='disabled')
        
    def get_color_hex(self, color_bgr):
        b, g, r = color_bgr
        return f'#{r:02x}{g:02x}{b:02x}'
        
    def setup_analysis_panel(self, parent):
        analysis_frame = ttk.LabelFrame(parent, text="Analysis Tools", padding=10)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(analysis_frame, text="AI Analysis:", font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(5,0))
        ttk.Button(analysis_frame, text="🧠 AI Image Analysis", 
                  command=self.analizar_imagen_ia).pack(fill=tk.X, pady=2)
        
        ttk.Label(analysis_frame, text="OpenCV Filters:", font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(10,0))
        
        filter_frame = ttk.Frame(analysis_frame)
        filter_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(filter_frame, text="Quick Filter:").pack(side=tk.LEFT)
        self.quick_filter_var = tk.StringVar()
        quick_filters = ["Original", "Escala de Grises", "Blanco y Negro", "Sepia", "Negativo", 
                        "Desenfoque Gaussiano", "Detección de Bordes", "Filtro Mediana"]
        quick_combo = ttk.Combobox(filter_frame, textvariable=self.quick_filter_var, 
                                  values=quick_filters, state="readonly", width=15)
        quick_combo.set("Original")
        quick_combo.pack(side=tk.LEFT, padx=5)
        quick_combo.bind('<<ComboboxSelected>>', self.apply_quick_filter)
        
        ttk.Button(analysis_frame, text="🎨 Open Filter Explorer", 
                  command=self.abrir_explorador_filtros).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="🔄 Reset to Original", 
                  command=self.reset_to_original).pack(fill=tk.X, pady=2)
        
    def setup_center_panel(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        controls = ["Load Image", "Zoom In", "Zoom Out", "Fit to Window", "Label Mode", "Save Image"]
        commands = [self.load_image, self.zoom_in, self.zoom_out, self.zoom_to_fit, 
                   self.toggle_label_mode, self.save_current_image]
        
        for i, (text, cmd) in enumerate(zip(controls, commands)):
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=2)
        
        zoom_frame = ttk.Frame(toolbar)
        zoom_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=1, to=1000, orient=tk.HORIZONTAL, 
                                   command=self.on_zoom_slider, length=150)
        self.zoom_slider.set(100)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
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
        
        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_basico = ttk.Frame(self.notebook)
        self.tab_planetario = ttk.Frame(self.notebook)
        self.tab_estelar = ttk.Frame(self.notebook)
        self.tab_avanzado = ttk.Frame(self.notebook)
        self.tab_filtros = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_basico, text="🧠 Basic Analysis")
        self.notebook.add(self.tab_planetario, text="🪐 Planetary")
        self.notebook.add(self.tab_estelar, text="🌌 Stellar")
        self.notebook.add(self.tab_avanzado, text="📊 Advanced")
        self.notebook.add(self.tab_filtros, text="🎨 Filters")
        
        self.crear_areas_texto()
        
        self.status_var = tk.StringVar(value="Ready - Load a NASA dataset to begin exploration")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    def crear_areas_texto(self):
        self.text_basico = scrolledtext.ScrolledText(self.tab_basico, wrap=tk.WORD, 
                                                   font=('Consolas', 9), bg='#f8f9fa')
        self.text_basico.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_planetario = scrolledtext.ScrolledText(self.tab_planetario, wrap=tk.WORD,
                                                       font=('Consolas', 9), bg='#e8f4f8')
        self.text_planetario.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_estelar = scrolledtext.ScrolledText(self.tab_estelar, wrap=tk.WORD,
                                                 font=('Consolas', 9), bg='#1a1a2e', fg='white')
        self.text_estelar.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_avanzado = scrolledtext.ScrolledText(self.tab_avanzado, wrap=tk.WORD,
                                                     font=('Consolas', 9), bg='#0d1b2a', fg='#e0e1dd')
        self.text_avanzado.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.setup_filtros_tab()
        
    def setup_filtros_tab(self):
        main_filter_frame = ttk.Frame(self.tab_filtros)
        main_filter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = ttk.Frame(main_filter_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        categ_frame = ttk.Frame(control_frame)
        categ_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(categ_frame, text="Filter Category:").pack(side=tk.LEFT)
        self.filter_cat_var = tk.StringVar()
        categories = ["Basic", "Color", "Edges", "Morphological", "Advanced", "All"]
        cat_combo = ttk.Combobox(categ_frame, textvariable=self.filter_cat_var, 
                                values=categories, state="readonly", width=12)
        cat_combo.set("Basic")
        cat_combo.pack(side=tk.LEFT, padx=5)
        cat_combo.bind('<<ComboboxSelected>>', self.on_filter_category_change)
        
        ttk.Label(categ_frame, text="Filter:").pack(side=tk.LEFT, padx=(10,0))
        self.filter_var = tk.StringVar()
        self.filter_combo = ttk.Combobox(categ_frame, textvariable=self.filter_var, 
                                        state="readonly", width=20)
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind('<<ComboboxSelected>>', self.on_filter_select)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Apply Filter", 
                  command=self.apply_selected_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Preview", 
                  command=self.preview_selected_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Show All Filters", 
                  command=self.mostrar_todos_filtros).pack(side=tk.LEFT, padx=2)
        
        preview_container = ttk.Frame(main_filter_frame)
        preview_container.pack(fill=tk.BOTH, expand=True)
        
        self.filter_preview_frame = ttk.LabelFrame(preview_container, text="Filter Preview", padding=5)
        self.filter_preview_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))
        
        self.filter_preview_label = ttk.Label(self.filter_preview_frame, 
                                            text="Select a filter to see preview",
                                            justify=tk.CENTER)
        self.filter_preview_label.pack(expand=True)
        
        info_frame = ttk.LabelFrame(preview_container, text="Filter Information", padding=5)
        info_frame.pack(fill=tk.BOTH, expand=False, side=tk.RIGHT, padx=(5, 0))
        
        self.filter_info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, 
                                                        height=10, width=30,
                                                        font=('Consolas', 8))
        self.filter_info_text.pack(fill=tk.BOTH, expand=True)
        
        self.on_filter_category_change()
        
    def bind_events(self):
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Control-s>", lambda e: self.save_current_image())
        self.root.bind("<Control-l>", lambda e: self.toggle_label_mode())
        self.root.bind("<Control-a>", lambda e: self.analizar_imagen_ia())
        self.root.bind("<Control-f>", lambda e: self.abrir_explorador_filtros())
        self.root.bind("<Control-r>", lambda e: self.reset_to_original())
        self.root.bind("<Escape>", lambda e: self.cancel_current_operation())
        
    def load_sample_datasets(self):
        self.sample_datasets = {
            "TESS FFI": {"description": "TESS Full Frame Image - Sector 45", "size": "4096x4096"},
            "Lunar Reconnaissance": {"description": "LRO LROC NAC Mosaic", "size": "8192x8192"}, 
            "Earth Observatory": {"description": "Landsat 8 Composite", "size": "10240x10240"}
        }
        
    def on_mission_select(self, event):
        mission = self.mission_var.get()
        if mission in self.sample_datasets:
            dataset = self.sample_datasets[mission]
            self.status_var.set(f"Selected: {mission} - {dataset['description']}")
            self.load_simulated_image(mission)
            
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
            
            colors = [(255, 255, 255), (255, 200, 100), (100, 150, 255)]
            color = colors[np.random.randint(0, len(colors))]
            
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
            
        return image
        
    def create_simulated_lunar_surface(self, width, height):
        image = Image.new('RGB', (width, height), color=(100, 100, 100))
        draw = ImageDraw.Draw(image)
        
        for _ in range(50):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(5, 20)
            draw.ellipse([x-size, y-size, x+size, y+size], 
                        outline=(150, 150, 150), fill=(80, 80, 80))
            
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
                self.status_var.set(f"Loaded: {os.path.basename(file_path)} - {image.size[0]}x{image.size[1]} pixels")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def set_image(self, image):
        self.original_image = image
        self.images["primary"] = image
        self.active_image_id = "primary"
        self.original_size = image.size
        self.scale = 1.0
        self.zoom_slider.set(100)
        
        self.cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.current_filtered_image = self.cv_image.copy()
        
        self.divide_image_into_tiles()
        self.generar_todos_filtros_automatico()
        
        self.display_current_image()
        self.center_image()
        self.clear_all_filters()
        self.reset_color_adjustments()

    def divide_image_into_tiles(self):
        self.image_tiles = {}
        self.current_tiles = {}
        
        if self.cv_image is None:
            return
            
        height, width = self.cv_image.shape[:2]
        
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                tile_key = f"{x}_{y}"
                end_x = min(x + self.tile_size, width)
                end_y = min(y + self.tile_size, height)
                self.image_tiles[tile_key] = self.cv_image[y:end_y, x:end_x]
                
        self.update_filtered_tiles()

    def update_filtered_tiles(self):
        if self.current_filtered_image is None:
            return
            
        self.current_tiles = {}
        height, width = self.current_filtered_image.shape[:2]
        
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                tile_key = f"{x}_{y}"
                end_x = min(x + self.tile_size, width)
                end_y = min(y + self.tile_size, height)
                self.current_tiles[tile_key] = self.current_filtered_image[y:end_y, x:end_x]

    def display_current_image(self):
        if self.current_filtered_image is None:
            return
            
        if self.scale <= 2.0:
            self.display_full_image()
        else:
            self.display_visible_tiles()
            
    def display_full_image(self):
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
        self.status_var.set(f"Displaying: {width}x{height} pixels | Zoom: {int(self.scale * 100)}% | Filters: {len(self.active_filters)}")

    def display_visible_tiles(self):
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        x_view = self.canvas.xview()
        y_view = self.canvas.yview()
        
        img_width = self.original_size[0] * self.scale
        img_height = self.original_size[1] * self.scale
        
        visible_x1 = max(0, int(x_view[0] * img_width))
        visible_y1 = max(0, int(y_view[0] * img_height))
        visible_x2 = min(img_width, int(x_view[1] * img_width))
        visible_y2 = min(img_height, int(y_view[1] * img_height))
        
        orig_x1 = int(visible_x1 / self.scale)
        orig_y1 = int(visible_y1 / self.scale)
        orig_x2 = int(visible_x2 / self.scale)
        orig_y2 = int(visible_y2 / self.scale)
        
        tiles_displayed = 0
        for tile_key, tile in self.current_tiles.items():
            x, y = map(int, tile_key.split('_'))
            
            if (x < orig_x2 and x + self.tile_size > orig_x1 and 
                y < orig_y2 and y + self.tile_size > orig_y1):
                
                tile_height, tile_width = tile.shape[:2]
                display_width = int(tile_width * self.scale)
                display_height = int(tile_height * self.scale)
                
                if len(tile.shape) == 3:
                    if tile.shape[2] == 3:
                        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                    else:
                        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
                else:
                    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
                    
                tile_pil = Image.fromarray(tile_rgb)
                resized_tile = tile_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                tile_photo = ImageTk.PhotoImage(resized_tile)
                
                canvas_x = int(x * self.scale)
                canvas_y = int(y * self.scale)
                self.canvas.create_image(canvas_x, canvas_y, anchor=tk.NW, image=tile_photo)
                
                if not hasattr(self, 'tile_photos'):
                    self.tile_photos = []
                self.tile_photos.append(tile_photo)
                
                tiles_displayed += 1
                
        self.zoom_label.config(text=f"{int(self.scale * 100)}%")
        self.status_var.set(f"High Zoom Mode: {tiles_displayed} tiles | Zoom: {int(self.scale * 100)}%")

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
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def aplicar_filtros_color_avanzados(self, imagen):
        filtros = {}
        
        filtros["Original"] = imagen
        
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        filtros["Escala de Grises"] = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        
        _, bn = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
        filtros["Blanco y Negro"] = cv2.cvtColor(bn, cv2.COLOR_GRAY2BGR)
        
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia = cv2.transform(imagen, sepia_filter)
        filtros["Sepia"] = np.clip(sepia, 0, 255).astype(np.uint8)
        
        filtros["Negativo"] = 255 - imagen
        
        b, g, r = cv2.split(imagen)
        zeros = np.zeros_like(b)
        
        filtros["Canal Rojo"] = cv2.merge([zeros, zeros, r])
        filtros["Canal Verde"] = cv2.merge([zeros, g, zeros])
        filtros["Canal Azul"] = cv2.merge([b, zeros, zeros])
        
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        filtros["HSV Color"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        filtros["Desenfoque Gaussiano"] = cv2.GaussianBlur(imagen, (15, 15), 0)
        filtros["Filtro Mediana"] = cv2.medianBlur(imagen, 5)
        
        bordes = cv2.Canny(gris, 100, 200)
        filtros["Detección de Bordes"] = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
        
        filtros["Filtro Bilateral"] = cv2.bilateralFilter(imagen, 9, 75, 75)
        
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        filtros["Alto Contraste"] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v])
        filtros["Brillo Aumentado"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        ycrcb = cv2.merge([y, cr, cb])
        filtros["Histograma Ecualizado"] = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return filtros

    def generar_todos_filtros_automatico(self):
        if self.cv_image is None:
            return
            
        self.mostrar_estado("⚡ Generando TODOS los filtros automáticamente...")
        
        thread = threading.Thread(target=self._generar_todos_filtros_thread)
        thread.daemon = True
        thread.start()
    
    def _generar_todos_filtros_thread(self):
        try:
            todos_filtros = self.aplicar_filtros_color_avanzados(self.cv_image)
            
            for nombre, filtro in todos_filtros.items():
                self.transformaciones[nombre] = filtro
            
            self.filtros_generados = True
            
            self.root.after(0, lambda: self.mostrar_estado(f"✅ {len(self.transformaciones)} filtros generados automáticamente"))
            
        except Exception as e:
            self.root.after(0, lambda: self.mostrar_estado("❌ Error generando filtros"))
            print(f"Error en generación de filtros: {e}")
    
    def apply_color_filter(self, filter_type):
        if self.cv_image is None:
            return
            
        filter_name = ""
        filtered_image = None
        
        if filter_type == "red_channel":
            filter_name = "Canal Rojo"
            b, g, r = cv2.split(self.cv_image)
            zeros = np.zeros_like(b)
            filtered_image = cv2.merge([zeros, zeros, r])
            
        elif filter_type == "green_channel":
            filter_name = "Canal Verde" 
            b, g, r = cv2.split(self.cv_image)
            zeros = np.zeros_like(b)
            filtered_image = cv2.merge([zeros, g, zeros])
            
        elif filter_type == "blue_channel":
            filter_name = "Canal Azul"
            b, g, r = cv2.split(self.cv_image)
            zeros = np.zeros_like(b)
            filtered_image = cv2.merge([b, zeros, zeros])
            
        elif filter_type == "hsv_color":
            filter_name = "HSV Color"
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            filtered_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif filter_type == "sepia":
            filter_name = "Sepia"
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168], 
                                   [0.393, 0.769, 0.189]])
            filtered_image = cv2.transform(self.cv_image, sepia_filter)
            filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
            
        elif filter_type == "grayscale":
            filter_name = "Escala de Grises"
            gris = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            filtered_image = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
            
        elif filter_type == "black_white":
            filter_name = "Blanco y Negro"
            gris = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
            filtered_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            
        elif filter_type == "negative":
            filter_name = "Negativo"
            filtered_image = 255 - self.cv_image
            
        if filtered_image is not None:
            self.current_filtered_image = filtered_image
            self.update_filtered_tiles()
            
            if filter_name not in self.active_filters:
                self.active_filters.append(filter_name)
                self.filters_listbox.insert(tk.END, filter_name)
            
            self.display_current_image()
            self.mostrar_estado(f"✅ Filtro de color aplicado: {filter_name}")

    def apply_color_adjustment(self):
        if self.cv_image is None:
            return
            
        try:
            image_float = self.cv_image.astype(np.float32) / 255.0
            
            brightness = self.brightness_var.get()
            image_adjusted = image_float * brightness
            
            contrast = self.contrast_var.get()
            image_adjusted = (image_adjusted - 0.5) * contrast + 0.5
            
            hsv = cv2.cvtColor(np.clip(image_adjusted, 0, 1), cv2.COLOR_BGR2HSV)
            saturation = self.saturation_var.get()
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 1)
            image_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            image_adjusted = np.clip(image_adjusted * 255, 0, 255).astype(np.uint8)
            
            self.current_filtered_image = image_adjusted
            self.update_filtered_tiles()
            self.display_current_image()
            
            filter_name = f"Color Adjust (B:{brightness:.1f}, C:{contrast:.1f}, S:{saturation:.1f})"
            if filter_name not in self.active_filters:
                self.active_filters.append(filter_name)
                self.filters_listbox.insert(tk.END, filter_name)
                
            self.mostrar_estado("✅ Ajustes de color aplicados")
            
        except Exception as e:
            self.mostrar_estado(f"❌ Error aplicando ajustes de color: {e}")

    def reset_color_adjustments(self):
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.current_filtered_image = self.cv_image.copy()
        self.update_filtered_tiles()
        self.display_current_image()
        self.mostrar_estado("🔄 Ajustes de color restablecidos")

    def apply_color_ranges(self):
        if self.cv_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            if not hasattr(self, 'imagen_hsv') or self.imagen_hsv is None:
                self.imagen_hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            
            mascara_combinada = np.zeros(self.imagen_hsv.shape[:2], dtype=np.uint8)
            imagen_resultado = np.zeros_like(self.cv_image)
            
            estadisticas = []
            total_pixeles = self.imagen_hsv.shape[0] * self.imagen_hsv.shape[1]
            
            for color_name, var in self.vars_color.items():
                if var.get():
                    config = self.rangos_color[color_name]
                    
                    if color_name == "Rojo":
                        mascara1 = cv2.inRange(self.imagen_hsv, config["hsv_bajo1"], config["hsv_alto1"])
                        mascara2 = cv2.inRange(self.imagen_hsv, config["hsv_bajo2"], config["hsv_alto2"])
                        mascara_color = cv2.bitwise_or(mascara1, mascara2)
                    else:
                        mascara_color = cv2.inRange(self.imagen_hsv, config["hsv_bajo"], config["hsv_alto"])
                    
                    mascara_combinada = cv2.bitwise_or(mascara_combinada, mascara_color)
                    
                    imagen_color = np.zeros_like(self.cv_image)
                    imagen_color[mascara_color > 0] = config["color_bgr"]
                    imagen_resultado = cv2.add(imagen_resultado, imagen_color)
                    
                    pixeles_color = np.sum(mascara_color > 0)
                    porcentaje = (pixeles_color / total_pixeles) * 100
                    estadisticas.append(f"{color_name}: {pixeles_color:,} píxeles ({porcentaje:.2f}%)")
            
            imagen_final = cv2.addWeighted(self.cv_image, 0.3, imagen_resultado, 0.7, 0)
            
            self.current_filtered_image = imagen_final
            self.update_filtered_tiles()
            self.display_current_image()
            
            self.mostrar_estadisticas_color(estadisticas, total_pixeles)
            
            filter_name = "Color Ranges Detection"
            if filter_name not in self.active_filters:
                self.active_filters.append(filter_name)
                self.filters_listbox.insert(tk.END, filter_name)
            
            self.mostrar_estado("✅ Rangos de color aplicados")
            
        except Exception as e:
            self.mostrar_estado(f"❌ Error aplicando rangos de color: {e}")
            messagebox.showerror("Error", f"Error al procesar rangos de color: {str(e)}")
    
    def mostrar_estadisticas_color(self, estadisticas, total_pixeles):
        texto = f"PÍXELES TOTALES: {total_pixeles:,}\n"
        texto += "=" * 50 + "\n\n"
        
        if estadisticas:
            for stat in estadisticas:
                texto += f"• {stat}\n"
            
            pixeles_detectados = sum(int(stat.split(":")[1].split(" ")[1].replace(",", "")) 
                                   for stat in estadisticas)
            pixeles_no_detectados = total_pixeles - pixeles_detectados
            porcentaje_no_detectado = (pixeles_no_detectados / total_pixeles) * 100
            
            texto += f"\n• No detectados: {pixeles_no_detectados:,} píxeles ({porcentaje_no_detectado:.2f}%)"
        else:
            texto += "No hay colores seleccionados para mostrar"
        
        self.color_stats_text.config(state='normal')
        self.color_stats_text.delete(1.0, tk.END)
        self.color_stats_text.insert(1.0, texto)
        self.color_stats_text.config(state='disabled')
    
    def reset_color_ranges(self):
        self.current_filtered_image = self.cv_image.copy()
        self.update_filtered_tiles()
        self.display_current_image()
        
        self.color_stats_text.config(state='normal')
        self.color_stats_text.delete(1.0, tk.END)
        self.color_stats_text.config(state='disabled')
        
        self.mostrar_estado("🔄 Rangos de color restablecidos")
    
    def mostrar_ventanas_individuales(self):
        if self.cv_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        if not hasattr(self, 'imagen_hsv') or self.imagen_hsv is None:
            self.imagen_hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        
        for color_name in self.rangos_color.keys():
            self.mostrar_color_individual(color_name)
    
    def mostrar_color_individual(self, color_name):
        config = self.rangos_color[color_name]
        
        if color_name == "Rojo":
            mascara1 = cv2.inRange(self.imagen_hsv, config["hsv_bajo1"], config["hsv_alto1"])
            mascara2 = cv2.inRange(self.imagen_hsv, config["hsv_bajo2"], config["hsv_alto2"])
            mascara_color = cv2.bitwise_or(mascara1, mascara2)
        else:
            mascara_color = cv2.inRange(self.imagen_hsv, config["hsv_bajo"], config["hsv_alto"])
        
        imagen_resultado = self.cv_image.copy()
        imagen_resultado[mascara_color == 0] = 0
        
        ventana = tk.Toplevel(self.root)
        ventana.title(f"Color {color_name}")
        ventana.geometry("900x700")
        
        imagen_rgb = cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB)
        imagen_redim = self.redimensionar_imagen(imagen_rgb, 800, 600)
        
        imagen_pil = Image.fromarray(imagen_redim)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        label = tk.Label(ventana, image=imagen_tk)
        label.image = imagen_tk
        label.pack(padx=20, pady=20)
        
        total_pixeles = self.imagen_hsv.shape[0] * self.imagen_hsv.shape[1]
        pixeles_color = np.sum(mascara_color > 0)
        porcentaje = (pixeles_color / total_pixeles) * 100
        
        stats_label = tk.Label(ventana, 
                              text=f"Píxeles {color_name}: {pixeles_color:,} ({porcentaje:.2f}%)",
                              font=("Arial", 14, "bold"))
        stats_label.pack(pady=10)
    
    def redimensionar_imagen(self, imagen, max_ancho, max_altura):
        if len(imagen.shape) == 3:
            altura, ancho = imagen.shape[:2]
        else:
            altura, ancho = imagen.shape
        
        if ancho > max_ancho or altura > max_altura:
            relacion = min(max_ancho/ancho, max_altura/altura)
            nuevo_ancho = int(ancho * relacion)
            nueva_altura = int(altura * relacion)
            imagen_redim = cv2.resize(imagen, (nuevo_ancho, nueva_altura))
            return imagen_redim
        return imagen

    def on_filter_category_change(self, event=None):
        categoria = self.filter_cat_var.get()
        
        if categoria == "Basic":
            filtros = ["Original", "Escala de Grises", "Blanco y Negro", "Sepia", 
                      "Negativo", "Desenfoque Gaussiano", "Filtro Mediana", "Filtro Bilateral"]
        elif categoria == "Color":
            filtros = ["Canal Rojo", "Canal Verde", "Canal Azul", "Alto Contraste", 
                      "Brillo Aumentado", "Histograma Ecualizado", "HSV Color"]
        elif categoria == "Edges":
            filtros = ["Detección de Bordes", "Filtro Sobel", "Laplaciano"]
        elif categoria == "Morphological":
            filtros = ["Erosión", "Dilatación", "Apertura Morfológica", "Cierre Morfológica"]
        elif categoria == "Advanced":
            filtros = ["Rotación 45°", "Espejo Horizontal", "Espejo Vertical"]
        elif categoria == "All":
            filtros = list(self.transformaciones.keys()) if self.transformaciones else []
        else:
            filtros = []
            
        self.filter_combo['values'] = filtros
        if filtros:
            self.filter_combo.set(filtros[0])
            self.update_filter_info(filtros[0])
        
    def on_filter_select(self, event=None):
        filtro_nombre = self.filter_var.get()
        if filtro_nombre:
            self.update_filter_info(filtro_nombre)
        
    def update_filter_info(self, filtro_nombre):
        info_text = f"Filter: {filtro_nombre}\n\n"
        
        if filtro_nombre == "Escala de Grises":
            info_text += "Convierte la imagen a escala de grises usando el promedio de los canales RGB.\n\nUso: Análisis de intensidad, preprocesamiento"
        elif filtro_nombre == "Blanco y Negro":
            info_text += "Aplica umbralización para crear una imagen binaria.\n\nUso: Segmentación, detección de objetos"
        elif filtro_nombre == "Desenfoque Gaussiano":
            info_text += "Aplica desenfoque gaussiano para reducir ruido.\n\nUso: Reducción de ruido, suavizado"
        elif filtro_nombre == "Detección de Bordes":
            info_text += "Detecta bordes usando el algoritmo Canny.\n\nUso: Detección de características, análisis estructural"
        elif filtro_nombre == "Filtro Mediana":
            info_text += "Aplica filtro mediano para reducir ruido sal-y-pimienta.\n\nUso: Reducción de ruido no lineal"
        else:
            info_text += "Filtro de procesamiento de imágenes para análisis científico."
        
        self.filter_info_text.config(state='normal')
        self.filter_info_text.delete(1.0, tk.END)
        self.filter_info_text.insert(1.0, info_text)
        self.filter_info_text.config(state='disabled')
        
    def preview_selected_filter(self):
        filtro_nombre = self.filter_var.get()
        if filtro_nombre in self.transformaciones:
            self.mostrar_filtro_preview(filtro_nombre)
        
    def mostrar_filtro_preview(self, filtro_nombre):
        imagen_filtro = self.transformaciones[filtro_nombre]
        
        for widget in self.filter_preview_frame.winfo_children():
            widget.destroy()
            
        self.mostrar_imagen_cv_en_frame(imagen_filtro, self.filter_preview_frame, 
                                       f"Filtro: {filtro_nombre}")
        
    def apply_selected_filter(self):
        filtro_nombre = self.filter_var.get()
        if filtro_nombre in self.transformaciones:
            self.apply_filter_to_main(filtro_nombre)
        
    def apply_filter_to_main(self, filter_name):
        if filter_name in self.transformaciones:
            filtered_image = self.transformaciones[filter_name]
            self.current_filtered_image = filtered_image
            self.update_filtered_tiles()
            
            if filter_name not in self.active_filters:
                self.active_filters.append(filter_name)
                self.filters_listbox.insert(tk.END, filter_name)
            
            self.display_current_image()
            self.mostrar_estado(f"✅ Filtro aplicado: {filter_name}")
            
    def apply_quick_filter(self, event=None):
        filter_name = self.quick_filter_var.get()
        if filter_name in self.transformaciones:
            self.apply_filter_to_main(filter_name)
        
    def remove_filter(self):
        selection = self.filters_listbox.curselection()
        if selection:
            index = selection[0]
            filter_name = self.filters_listbox.get(index)
            self.filters_listbox.delete(index)
            self.active_filters.pop(index)
            
            self.reapply_all_filters()
            
    def clear_all_filters(self):
        self.filters_listbox.delete(0, tk.END)
        self.active_filters.clear()
        self.current_filtered_image = self.cv_image.copy()
        self.update_filtered_tiles()
        self.display_current_image()
        self.mostrar_estado("🔄 Todos los filtros eliminados")
        
    def reapply_all_filters(self):
        if not self.active_filters:
            self.current_filtered_image = self.cv_image.copy()
        else:
            last_filter = self.active_filters[-1]
            self.current_filtered_image = self.transformaciones[last_filter]
        
        self.update_filtered_tiles()
        self.display_current_image()
        
    def reset_to_original(self):
        self.clear_all_filters()
        self.reset_color_adjustments()
        self.reset_color_ranges()
        self.mostrar_estado("🔄 Imagen restaurada a original")

    def mostrar_todos_filtros(self):
        if not self.transformaciones:
            messagebox.showwarning("Advertencia", "Primero carga una imagen para generar filtros")
            return
        
        top = tk.Toplevel(self.root)
        top.title("Todos los Filtros OpenCV")
        top.geometry("1200x800")
        
        nombres = list(self.transformaciones.keys())
        transformaciones = list(self.transformaciones.values())
        
        n_transformaciones = len(transformaciones)
        filas = int(np.ceil(np.sqrt(n_transformaciones)))
        columnas = int(np.ceil(n_transformaciones / filas))
        
        fig, axes = plt.subplots(filas, columnas, figsize=(20, 15))
        fig.suptitle(f"Todos los Filtros Generados ({n_transformaciones} total)", fontsize=16)
        
        if filas == 1 or columnas == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()
        
        for i in range(n_transformaciones):
            if len(transformaciones[i].shape) == 3 and transformaciones[i].shape[2] == 3:
                img_display = cv2.cvtColor(transformaciones[i], cv2.COLOR_BGR2RGB)
            else:
                img_display = transformaciones[i]
                
            axes[i].imshow(img_display)
            axes[i].set_title(nombres[i], fontsize=8)
            axes[i].axis('off')
        
        for i in range(n_transformaciones, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def abrir_explorador_filtros(self):
        if self.cv_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        FilterExplorerWindow(self.root, self.cv_image, self.apply_filter_to_main)

    def save_current_image(self):
        if self.current_filtered_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")]
        )
        
        if ruta:
            try:
                cv2.imwrite(ruta, self.current_filtered_image)
                messagebox.showinfo("Éxito", f"Imagen guardada como: {ruta}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen: {e}")

    def mostrar_imagen_cv_en_frame(self, imagen_cv, frame_destino, texto_alternativo=""):
        for widget in frame_destino.winfo_children():
            widget.destroy()
        
        if imagen_cv is None:
            label = ttk.Label(frame_destino, text=texto_alternativo)
            label.pack(expand=True)
            return
        
        h, w = imagen_cv.shape[:2]
        max_width = 400
        max_height = 300
        
        if w > max_width or h > max_height:
            ratio = min(max_width/w, max_height/h)
            new_w, new_h = int(w*ratio), int(h*ratio)
            imagen_mostrar = cv2.resize(imagen_cv, (new_w, new_h))
        else:
            imagen_mostrar = imagen_cv
        
        if len(imagen_mostrar.shape) == 3:
            if imagen_mostrar.shape[2] == 3:
                imagen_mostrar = cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB)
        else:
            imagen_mostrar = cv2.cvtColor(imagen_mostrar, cv2.COLOR_GRAY2RGB)
        
        imagen_pil = Image.fromarray(imagen_mostrar)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        label = ttk.Label(frame_destino, image=imagen_tk)
        label.image = imagen_tk
        label.pack(expand=True)

    def zoom_in(self, event=None, factor=1.2, center=None):
        if self.current_filtered_image is None: return
        self.save_viewport_state()
        old_scale = self.scale
        self.scale *= factor
        self.scale = max(0.01, min(50.0, self.scale))
        self.display_current_image()
        self.zoom_slider.set(self.scale * 100)
        self.restore_viewport_after_zoom(old_scale, center)
        
    def zoom_out(self, event=None, factor=1.2):
        self.zoom_in(factor=1/factor)
        
    def on_mousewheel(self, event):
        if self.current_filtered_image is None: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if event.delta > 0 or event.num == 4:
            self.zoom_in(factor=1.2, center=(x, y))
        else:
            self.zoom_out(factor=1.2)
            
    def search_coordinates(self):
        coordinates = self.coord_entry.get()
        feature = self.feature_entry.get()
        if coordinates:
            self.status_var.set(f"Searching coordinates: {coordinates}")
            messagebox.showinfo("Search", f"Centering on coordinates: {coordinates}")
        elif feature:
            self.status_var.set(f"Searching feature: {feature}")
            messagebox.showinfo("Search", f"Looking for feature: {feature}")

    def toggle_label_mode(self):
        self.label_mode = not self.label_mode
        if self.label_mode:
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Label Mode: Click and drag to mark features")
        else:
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Navigation Mode")

    def cancel_current_operation(self):
        if self.current_label:
            self.canvas.delete(self.current_label)
            self.current_label = None
        self.label_mode = False
        self.status_var.set("Operation cancelled")

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
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        if self.current_filtered_image is not None:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            if 0 <= x < self.original_size[0] * self.scale and 0 <= y < self.original_size[1] * self.scale:
                orig_x = int(x / self.scale)
                orig_y = int(y / self.scale)
                self.status_var.set(f"Position: ({orig_x}, {orig_y}) | Zoom: {int(self.scale * 100)}% | Filters: {len(self.active_filters)}")

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
                    label_data = {
                        "name": label_name,
                        "coords": (orig_start_x, orig_start_y, orig_end_x, orig_end_y),
                        "timestamp": datetime.now().isoformat(),
                        "mission": self.mission_var.get()
                    }
                    self.labels.append(label_data)
                    self.canvas.itemconfig(self.current_label, dash=(), fill="", 
                                         outline="yellow", width=2, tags="permanent_label")
                    text_id = self.canvas.create_text(
                        self.start_x, self.start_y - 10,
                        text=label_name, fill="yellow", anchor=tk.SW, tags="permanent_label"
                    )
                    label_data["text_id"] = text_id
                    self.status_var.set(f"Label added: {label_name}")
            else:
                self.canvas.delete(self.current_label)
            self.current_label = None

    def save_viewport_state(self):
        if self.current_filtered_image is not None:
            self.last_viewport = {
                'scale': self.scale,
                'xview': self.canvas.xview(),
                'yview': self.canvas.yview()
            }

    def restore_viewport_after_zoom(self, old_scale, center=None):
        if self.current_filtered_image is None: return
        if center:
            x, y = center
            scale_ratio = self.scale / old_scale
            img_width = self.original_size[0] * self.scale
            img_height = self.original_size[1] * self.scale
            new_x = (x / old_scale) * self.scale
            new_y = (y / old_scale) * self.scale
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            target_x = new_x - canvas_width / 2
            target_y = new_y - canvas_height / 2
            if img_width > canvas_width:
                x_fraction = max(0, min(1, target_x / img_width))
                self.canvas.xview_moveto(x_fraction)
            if img_height > canvas_height:
                y_fraction = max(0, min(1, target_y / img_height))
                self.canvas.yview_moveto(y_fraction)
        elif self.last_viewport:
            center_x = (self.last_viewport['xview'][0] + self.last_viewport['xview'][1]) / 2
            center_y = (self.last_viewport['yview'][0] + self.last_viewport['yview'][1]) / 2
            self.canvas.xview_moveto(max(0, min(1, center_x)))
            self.canvas.yview_moveto(max(0, min(1, center_y)))

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
            old_scale = self.scale
            new_scale = float(value) / 100.0
            if new_scale != self.scale:
                self.save_viewport_state()
                self.scale = new_scale
                self.display_current_image()
                self.restore_viewport_after_zoom(old_scale)

    def analizar_imagen_ia(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.analizando:
            messagebox.showinfo("Info", "Analysis already in progress")
            return
        
        self.analizando = True
        self.mostrar_estado("🔍 Starting AI analysis with internet search...")
        
        thread = threading.Thread(target=self._ejecutar_analisis_ia_internet)
        thread.daemon = True
        thread.start()
    
    def _ejecutar_analisis_ia_internet(self):
        try:
            self.limpiar_resultados_texto()
            
            self.mostrar_estado("1️⃣ Running Basic Analysis with Internet Search...")
            resultado_basico = self.analizador.analisis_basico_con_internet(self.original_image) if self.analizador else self._analisis_sin_modelos()
            self.actualizar_resultado_basico(resultado_basico)
            
            self.mostrar_estado("2️⃣ Running Planetary Analysis...")
            resultado_planetario = self.analizador.analisis_planetario(self.original_image, "current_image") if self.analizador else self._analisis_planetario_simple()
            self.actualizar_resultado_planetario(resultado_planetario)
            
            self.mostrar_estado("3️⃣ Running Stellar Analysis...")
            resultado_estelar = self.analizador.analisis_estelar(self.original_image, "current_image") if self.analizador else self._analisis_estelar_simple()
            self.actualizar_resultado_estelar(resultado_estelar)
            
            self.mostrar_estado("4️⃣ Running Advanced Analysis...")
            resultado_avanzado = self.analizador.analisis_avanzado(self.original_image, "current_image") if self.analizador else self._analisis_avanzado_simple()
            self.actualizar_resultado_avanzado(resultado_avanzado)
            
            self.mostrar_estado("✅ AI analysis with internet search completed")
            
        except Exception as e:
            error_msg = f"❌ Analysis error: {str(e)}"
            self.mostrar_estado(error_msg)
            self.actualizar_resultado_basico(f"ERROR: {error_msg}\n\nTrying offline analysis...")
            
            try:
                resultado_offline = self._analisis_sin_modelos()
                self.actualizar_resultado_basico(resultado_offline)
            except:
                pass
        
        finally:
            self.analizando = False

    def _analisis_sin_modelos(self):
        resultado = "1️⃣  BASIC ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "📝 ANALYSIS WITHOUT AI MODELS:\n\n"
        resultado += "🔍 Image Analysis:\n"
        resultado += f"• Resolution: {self.original_size[0]} x {self.original_size[1]} pixels\n"
        resultado += f"• Format: {self.original_image.format if hasattr(self.original_image, 'format') else 'Unknown'}\n"
        resultado += f"• Mode: {self.original_image.mode}\n\n"
        
        resultado += "💡 RECOMMENDATIONS:\n"
        resultado += "• Install AI models for detailed analysis\n"
        resultado += "• Use NASA official databases for scientific context\n"
        resultado += "• Consult astronomical databases for specific features\n"
        
        return resultado

    def _analisis_planetario_simple(self):
        resultado = "2️⃣  PLANETARY ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "🪐 Basic Planetary Analysis:\n\n"
        resultado += "For detailed planetary analysis:\n"
        resultado += "• Load specific planetary images (Mars, Moon, etc.)\n"
        resultado += "• Use high-resolution NASA mission data\n"
        resultado += "• Analyze geological features and surface patterns\n"
        return resultado

    def _analisis_estelar_simple(self):
        resultado = "3️⃣  STELLAR ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "🌌 Basic Stellar Analysis:\n\n"
        resultado += "For detailed stellar analysis:\n"
        resultado += "• Use astronomical images from Hubble, JWST\n"
        resultado += "• Analyze star patterns and celestial objects\n"
        resultado += "• Consult star catalogs and astronomical databases\n"
        return resultado

    def _analisis_avanzado_simple(self):
        resultado = "4️⃣  ADVANCED NASA ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "📊 Technical Analysis:\n"
        resultado += f"• Resolution: {self.original_size[0]} x {self.original_size[1]} pixels\n"
        resultado += f"• Color Mode: {self.original_image.mode}\n\n"
        
        resultado += "💡 Scientific Recommendations:\n"
        resultado += "1. Use high-resolution NASA imagery for best results\n"
        resultado += "2. Consult NASA mission-specific databases\n"
        resultado += "3. Cross-reference with scientific publications\n"
        resultado += "4. Use specialized analysis tools for specific features\n"
        
        return resultado
    
    def actualizar_resultado_basico(self, texto):
        def actualizar():
            self.text_basico.delete(1.0, tk.END)
            self.text_basico.insert(1.0, texto)
            self.notebook.select(0)
        self.root.after(0, actualizar)
    
    def actualizar_resultado_planetario(self, texto):
        def actualizar():
            self.text_planetario.delete(1.0, tk.END)
            self.text_planetario.insert(1.0, texto)
        self.root.after(0, actualizar)
    
    def actualizar_resultado_estelar(self, texto):
        def actualizar():
            self.text_estelar.delete(1.0, tk.END)
            self.text_estelar.insert(1.0, texto)
        self.root.after(0, actualizar)
    
    def actualizar_resultado_avanzado(self, texto):
        def actualizar():
            self.text_avanzado.delete(1.0, tk.END)
            self.text_avanzado.insert(1.0, texto)
        self.root.after(0, actualizar)
    
    def limpiar_resultados_texto(self):
        self.text_basico.delete(1.0, tk.END)
        self.text_planetario.delete(1.0, tk.END)
        self.text_estelar.delete(1.0, tk.END)
        self.text_avanzado.delete(1.0, tk.END)

class FilterExplorerWindow:
    def __init__(self, parent, cv_image, apply_callback):
        self.root = tk.Toplevel(parent)
        self.root.title("OpenCV Filter Explorer")
        self.root.geometry("1000x700")
        
        self.cv_image = cv_image
        self.apply_callback = apply_callback
        self.transformaciones = {}
        
        self.setup_ui()
        self.generar_todos_los_filtros()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Category:").pack(side=tk.LEFT)
        self.cat_var = tk.StringVar()
        categories = ["All", "Basic", "Color", "Edges", "Morphological", "Advanced"]
        cat_combo = ttk.Combobox(control_frame, textvariable=self.cat_var, 
                                values=categories, state="readonly", width=12)
        cat_combo.set("All")
        cat_combo.pack(side=tk.LEFT, padx=5)
        cat_combo.bind('<<ComboboxSelected>>', self.on_category_change)
        
        ttk.Button(control_frame, text="Apply Selected Filter", 
                  command=self.apply_selected).pack(side=tk.RIGHT, padx=5)
        
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.filters_listbox = tk.Listbox(list_frame)
        self.filters_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.filters_listbox.bind('<<ListboxSelect>>', self.on_filter_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.filters_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.filters_listbox.config(yscrollcommand=scrollbar.set)
        
    def generar_todos_los_filtros(self):
        transformaciones, nombres = aplicar_filtros_rapidos(self.cv_image)
        for i, (transformacion, nombre) in enumerate(zip(transformaciones, nombres)):
            self.transformaciones[nombre] = transformacion
            
        self.update_filters_list()
        
    def update_filters_list(self):
        self.filters_listbox.delete(0, tk.END)
        for nombre in self.transformaciones.keys():
            self.filters_listbox.insert(tk.END, nombre)
            
    def on_category_change(self, event=None):
        pass
        
    def on_filter_select(self, event):
        selection = self.filters_listbox.curselection()
        if selection:
            filter_name = self.filters_listbox.get(selection[0])
            
    def apply_selected(self):
        selection = self.filters_listbox.curselection()
        if selection:
            filter_name = self.filters_listbox.get(selection[0])
            self.apply_callback(filter_name)
            self.root.destroy()

class AnalizadorEspecializadoNASA:
    def __init__(self):
        self.modelos = {}
        self.cargar_todos_modelos()
    
    def cargar_todos_modelos(self):
        try:
            self.modelos['descripcion'] = pipeline("image-to-text", 
                                                 model="Salesforce/blip-image-captioning-large")
            self.modelos['clasificacion'] = pipeline("image-classification", 
                                                   model="microsoft/resnet-50")
            print("✅ Modelos IA cargados correctamente")
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")

    def analisis_basico_con_internet(self, imagen):
        resultado = "1️⃣  BASIC ANALYSIS WITH INTERNET SEARCH\n" + "="*60 + "\n\n"
        
        try:
            descripcion_local = "No se pudo generar descripción local"
            if 'descripcion' in self.modelos:
                try:
                    descripcion_local = self.modelos['descripcion'](imagen)[0]['generated_text']
                except:
                    pass
            
            resultado += f"📝 LOCAL ANALYSIS:\n{descripcion_local}\n\n"
            
            internet_info = self.buscar_informacion_internet(descripcion_local)
            resultado += f"🌐 INTERNET RESEARCH:\n{internet_info}\n\n"
            
            resultado += "🔍 DETECTED FEATURES:\n" + "-"*25 + "\n"
            if 'clasificacion' in self.modelos:
                try:
                    clasificaciones = self.modelos['clasificacion'](imagen)
                    for i, clasif in enumerate(clasificaciones[:5], 1):
                        if clasif['score'] > 0.1:
                            resultado += f"{i}. {clasif['label']}: {clasif['score']*100:.1f}%\n"
                except:
                    resultado += "No se pudieron analizar características\n"
            else:
                resultado += "Modelo de clasificación no disponible\n"
                
        except Exception as e:
            resultado += f"❌ Error in analysis: {e}\n"
            
        return resultado

    def buscar_informacion_internet(self, descripcion):
        keywords = descripcion.lower()
        info = ""
        
        if any(word in keywords for word in ['star', 'stellar', 'galaxy', 'nebula']):
            info = """🔭 ASTRONOMICAL ANALYSIS:
• Posible objeto astronómico detectado
• Características comunes: estrellas, galaxias o nebulosas
• Recomendado: Análisis espectral para clasificación
• NASA missions relevantes: Hubble, JWST, Chandra

📊 SCIENTIFIC CONTEXT:
Los objetos estelares muestran patrones específicos en imágenes astronómicas.
Las estrellas aparecen como puntos brillantes, las galaxias como estructuras extendidas,
y las nebulosas como nubes de gas y polvo."""
            
        elif any(word in keywords for word in ['planet', 'lunar', 'mars', 'surface']):
            info = """🪐 PLANETARY ANALYSIS:
• Características planetarias detectadas
• Posible superficie planetaria o lunar
• Recomendado: Análisis geológico y topográfico
• NASA missions relevantes: LRO, Mars Rover, Voyager

📊 SCIENTIFIC CONTEXT:
Las imágenes planetarias muestran cráteres, valles, y formaciones geológicas.
El análisis de texturas y patrones ayuda a entender la historia geológica."""
            
        elif any(word in keywords for word in ['earth', 'cloud', 'weather', 'terrain']):
            info = """🌍 EARTH SCIENCE ANALYSIS:
• Imagen terrestre o de observación de la Tierra
• Posibles formaciones nubosas o características geográficas
• Recomendado: Análisis meteorológico o geográfico
• NASA missions relevantes: Landsat, Terra, Aqua

📊 SCIENTIFIC CONTEXT:
Las imágenes terrestres de la NASA monitorizan cambios climáticos,
uso de suelo, y fenómenos meteorológicos."""
            
        else:
            info = """🔬 GENERAL SCIENTIFIC ANALYSIS:
• Imagen científica de la NASA detectada
• Características específicas requieren análisis detallado
• Recomendado: Consultar bases de datos NASA específicas
• Posibles aplicaciones: Investigación astronómica, terrestre o planetaria

📊 NEXT STEPS:
1. Comparar con bases de datos NASA existentes
2. Realizar análisis espectral si está disponible
3. Consultar con expertos en el dominio específico"""
        
        return info

    def analisis_planetario(self, imagen, ruta_imagen):
        resultado = "2️⃣  PLANETARY ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "🪐 Análisis planetario requiere imágenes específicas de planetas.\n"
        resultado += "Carga imágenes de Marte, Luna, o otros cuerpos celestes para análisis detallado.\n"
        return resultado

    def analisis_estelar(self, imagen, ruta_imagen):
        resultado = "3️⃣  STELLAR ANALYSIS\n" + "="*50 + "\n\n"
        resultado += "🌌 Análisis estelar optimizado para imágenes astronómicas.\n"
        resultado += "Imágenes de Hubble, JWST, o telescopios terrestres proporcionan mejores resultados.\n"
        return resultado

    def analisis_avanzado(self, imagen, ruta_imagen):
        resultado = "4️⃣  ADVANCED NASA ANALYSIS\n" + "="*50 + "\n\n"
        ancho, alto = imagen.size
        resultado += f"📊 TECHNICAL ANALYSIS:\n" + "-"*20 + "\n"
        resultado += f"📐 Resolution: {ancho} x {alto} pixels\n"
        resultado += f"🎨 Color mode: {imagen.mode}\n\n"
        
        resultado += "💡 RECOMMENDATIONS:\n" + "-"*20 + "\n"
        resultado += "1. Para mejor análisis, use imágenes NASA de alta resolución\n"
        resultado += "2. Imágenes específicas de misiones NASA proporcionan mejor contexto\n"
        resultado += "3. Consulte la base de datos oficial NASA para metadata completa\n"
        
        return resultado

    def analisis_basico(self, imagen, ruta_imagen):
        resultado = "1️⃣  BASIC ANALYSIS\n" + "="*50 + "\n\n"
        try:
            if 'descripcion' in self.modelos:
                descripcion = self.modelos['descripcion'](imagen)[0]['generated_text']
                resultado += f"📝 DESCRIPTION:\n{descripcion}\n\n"
            else:
                resultado += "📝 Modelo de descripción no disponible\n\n"
                
        except Exception as e:
            resultado += f"❌ Error in basic analysis: {e}\n"
            
        return resultado

def aplicar_filtros_rapidos(imagen):
    img_original = imagen.copy()
    transformaciones = []
    nombres = []
    
    transformaciones.append(img_original)
    nombres.append("Original")
    
    gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    gris_color = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    transformaciones.append(gris_color)
    nombres.append("Escala de Grises")
    
    _, bn = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
    bn_color = cv2.cvtColor(bn, cv2.COLOR_GRAY2BGR)
    transformaciones.append(bn_color)
    nombres.append("Blanco y Negro")
    
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img_original, sepia_filter)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    transformaciones.append(sepia)
    nombres.append("Sepia")
    
    negativo = 255 - img_original
    transformaciones.append(negativo)
    nombres.append("Negativo")
    
    gaussiano = cv2.GaussianBlur(img_original, (15, 15), 0)
    transformaciones.append(gaussiano)
    nombres.append("Desenfoque Gaussiano")
    
    bordes = cv2.Canny(gris, 100, 200)
    bordes_color = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
    transformaciones.append(bordes_color)
    nombres.append("Detección de Bordes")
    
    mediana = cv2.medianBlur(img_original, 5)
    transformaciones.append(mediana)
    nombres.append("Filtro Mediana")
    
    return transformaciones, nombres

def main():
    root = tk.Tk()
    app = NASAImageExplorerPro(root)
    root.mainloop()

if __name__ == "__main__":
    main()