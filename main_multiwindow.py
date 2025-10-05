import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from PIL import Image
import cv2
import numpy as np

class NASAImageWindowManager:
    """Gestor principal de múltiples ventanas de análisis de imágenes NASA"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌌 NASA Multi-Window Image Explorer - Main Controller")
        self.root.geometry("500x600")
        
        # Diccionario para trackear ventanas abiertas
        self.windows = {}
        self.window_counter = 1
        
        # Variables de estado - INICIALIZAR PRIMERO
        self.status_var = tk.StringVar(value="Sistema listo - Crea una nueva ventana para comenzar")
        self.stats_vars = {
            'ventanas_activas': tk.StringVar(value="0"),
            'imagenes_cargadas': tk.StringVar(value="0"),
            'memoria_estimada': tk.StringVar(value="0 MB")
        }
        
        # Configuración de estilo
        self.setup_styles()
        
        # Interfaz principal
        self.setup_main_ui()
        
    def setup_styles(self):
        """Configurar estilos para la interfaz"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 12), foreground='#34495e')
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        
    def setup_main_ui(self):
        """Configurar la interfaz principal del gestor"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="🌌 NASA Multi-Window Explorer", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        subtitle_label = ttk.Label(main_frame, text="Gestor de Ventanas de Análisis Científico", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=5)
        
        # Separador
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=15)
        
        # Panel de control rápido
        self.setup_quick_controls(main_frame)
        
        # Lista de ventanas activas
        self.setup_windows_panel(main_frame)
        
        # Panel de estadísticas
        self.setup_stats_panel(main_frame)
        
        # Barra de estado
        self.setup_status_bar(main_frame)
        
    def setup_quick_controls(self, parent):
        """Panel de controles rápidos"""
        control_frame = ttk.LabelFrame(parent, text="🛠️ Controles Rápidos", padding=15)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Botones principales
        btn_grid = ttk.Frame(control_frame)
        btn_grid.pack(fill=tk.X)
        
        # Fila 1
        row1 = ttk.Frame(btn_grid)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Button(row1, text="🪟 Nueva Ventana Vacía", 
                  command=self.nueva_ventana_vacia,
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(row1, text="📁 Cargar Imagen en Nueva Ventana", 
                  command=self.nueva_ventana_con_imagen,
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Fila 2
        row2 = ttk.Frame(btn_grid)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Button(row2, text="🚀 Cargar Múltiples Imágenes", 
                  command=self.cargar_multiple_imagenes,
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(row2, text="🗑️ Cerrar Todas las Ventanas", 
                  command=self.cerrar_todas_ventanas,
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
    def setup_windows_panel(self, parent):
        """Panel de lista de ventanas activas"""
        windows_frame = ttk.LabelFrame(parent, text="📋 Ventanas Activas", padding=10)
        windows_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Frame de controles de ventana
        window_controls = ttk.Frame(windows_frame)
        window_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(window_controls, text="🔍 Enfocar Ventana", 
                  command=self.focus_selected_window).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(window_controls, text="❌ Cerrar Ventana", 
                  command=self.cerrar_ventana_seleccionada).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(window_controls, text="🔄 Actualizar Lista", 
                  command=self.actualizar_lista_ventanas).pack(side=tk.RIGHT, padx=2)
        
        # Lista de ventanas
        list_frame = ttk.Frame(windows_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview para mostrar información detallada de ventanas
        columns = ('ID', 'Estado', 'Imagen', 'Filtros')
        self.windows_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        # Configurar columnas
        self.windows_tree.heading('ID', text='ID Ventana')
        self.windows_tree.heading('Estado', text='Estado')
        self.windows_tree.heading('Imagen', text='Imagen')
        self.windows_tree.heading('Filtros', text='Filtros Activos')
        
        self.windows_tree.column('ID', width=80)
        self.windows_tree.column('Estado', width=100)
        self.windows_tree.column('Imagen', width=150)
        self.windows_tree.column('Filtros', width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.windows_tree.yview)
        self.windows_tree.configure(yscrollcommand=scrollbar.set)
        
        self.windows_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind eventos
        self.windows_tree.bind('<Double-1>', lambda e: self.focus_selected_window())
        
    def setup_stats_panel(self, parent):
        """Panel de estadísticas del sistema"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Estadísticas del Sistema", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Estadísticas en tiempo real
        ttk.Label(stats_grid, text="Ventanas activas:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(stats_grid, textvariable=self.stats_vars['ventanas_activas'], 
                 font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(stats_grid, text="Imágenes cargadas:").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Label(stats_grid, textvariable=self.stats_vars['imagenes_cargadas'],
                 font=('Arial', 10, 'bold')).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(stats_grid, text="Memoria estimada:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Label(stats_grid, textvariable=self.stats_vars['memoria_estimada'],
                 font=('Arial', 10, 'bold')).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        stats_grid.columnconfigure(1, weight=1)
        stats_grid.columnconfigure(3, weight=1)
        
    def setup_status_bar(self, parent):
        """Barra de estado"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, padding=5)
        status_bar.pack(fill=tk.X)
        
    def nueva_ventana_vacia(self):
        """Crear una nueva ventana de análisis vacía"""
        window_id = self.window_counter
        self.window_counter += 1
        
        # Crear ventana Toplevel
        new_window = tk.Toplevel(self.root)
        new_window.title(f"NASA Image Explorer - Ventana {window_id}")
        new_window.geometry("1400x900")
        
        # Importar y crear instancia del explorador NASA desde app.py
        try:
            from app import NASAImageExplorerPro
            app_instance = NASAImageExplorerPro(new_window)
            
            # Guardar referencia
            self.windows[window_id] = {
                'window': new_window,
                'instance': app_instance,
                'title': f"Ventana {window_id}",
                'image_loaded': False,
                'filters_count': 0
            }
            
            # Configurar protocolo de cierre
            new_window.protocol("WM_DELETE_WINDOW", 
                              lambda wid=window_id: self.cerrar_ventana(wid))
            
            # Actualizar interfaz
            self.actualizar_lista_ventanas()
            self.actualizar_estadisticas()
            
            self.status_var.set(f"✅ Ventana {window_id} creada - Lista para cargar imagen")
            
        except ImportError as e:
            messagebox.showerror("Error", f"No se pudo importar NASAImageExplorerPro desde app.py: {e}")
            new_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear ventana: {e}")
            new_window.destroy()
            
    def nueva_ventana_con_imagen(self):
        """Crear nueva ventana y cargar imagen inmediatamente"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen para Nueva Ventana",
            filetypes=[
                ("Imágenes NASA", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.fits"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            window_id = self.window_counter
            self.window_counter += 1
            
            # Crear ventana
            new_window = tk.Toplevel(self.root)
            filename = os.path.basename(file_path)
            new_window.title(f"NASA Explorer - {filename}")
            new_window.geometry("1400x900")
            
            try:
                from app import NASAImageExplorerPro
                app_instance = NASAImageExplorerPro(new_window)
                
                # Cargar la imagen en la nueva ventana usando el método existente
                app_instance.load_image(file_path)
                
                # Guardar referencia
                self.windows[window_id] = {
                    'window': new_window,
                    'instance': app_instance,
                    'title': f"Ventana {window_id} - {filename}",
                    'image_loaded': True,
                    'image_name': filename,
                    'filters_count': 0
                }
                
                # Configurar cierre
                new_window.protocol("WM_DELETE_WINDOW", 
                                  lambda wid=window_id: self.cerrar_ventana(wid))
                
                # Actualizar interfaz
                self.actualizar_lista_ventanas()
                self.actualizar_estadisticas()
                
                self.status_var.set(f"✅ Ventana {window_id} creada con {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al crear ventana: {e}")
                new_window.destroy()
                
    def cargar_multiple_imagenes(self):
        """Cargar múltiples imágenes en ventanas separadas"""
        file_paths = filedialog.askopenfilenames(
            title="Seleccionar Múltiples Imágenes",
            filetypes=[
                ("Imágenes NASA", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.fits"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_paths:
            self.status_var.set(f"🔄 Cargando {len(file_paths)} imágenes...")
            
            for i, file_path in enumerate(file_paths):
                # Usar after para no bloquear la interfaz
                self.root.after(i * 500, lambda fp=file_path: self.crear_ventana_con_imagen_delay(fp))
                
    def crear_ventana_con_imagen_delay(self, file_path):
        """Crear ventana con imagen con retardo para evitar bloqueo"""
        window_id = self.window_counter
        self.window_counter += 1
        
        new_window = tk.Toplevel(self.root)
        filename = os.path.basename(file_path)
        new_window.title(f"NASA Explorer - {filename}")
        new_window.geometry("1400x900")
        
        try:
            from app import NASAImageExplorerPro
            app_instance = NASAImageExplorerPro(new_window)
            app_instance.load_image(file_path)
            
            self.windows[window_id] = {
                'window': new_window,
                'instance': app_instance,
                'title': f"Ventana {window_id} - {filename}",
                'image_loaded': True,
                'image_name': filename,
                'filters_count': 0
            }
            
            new_window.protocol("WM_DELETE_WINDOW", 
                              lambda wid=window_id: self.cerrar_ventana(wid))
            
            self.actualizar_lista_ventanas()
            self.actualizar_estadisticas()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar {filename}: {e}")
            new_window.destroy()
            
    def focus_selected_window(self):
        """Enfocar la ventana seleccionada en la lista"""
        selection = self.windows_tree.selection()
        if selection:
            item = selection[0]
            window_id = int(self.windows_tree.item(item, 'values')[0])
            
            if window_id in self.windows:
                window_info = self.windows[window_id]
                try:
                    window_info['window'].lift()
                    window_info['window'].focus_force()
                    self.status_var.set(f"🔍 Ventana {window_id} enfocada")
                except:
                    self.status_var.set(f"❌ No se pudo enfocar ventana {window_id}")
                    
    def cerrar_ventana_seleccionada(self):
        """Cerrar la ventana seleccionada en la lista"""
        selection = self.windows_tree.selection()
        if selection:
            item = selection[0]
            window_id = int(self.windows_tree.item(item, 'values')[0])
            self.cerrar_ventana(window_id)
            
    def cerrar_ventana(self, window_id):
        """Cerrar una ventana específica"""
        if window_id in self.windows:
            try:
                self.windows[window_id]['window'].destroy()
                del self.windows[window_id]
                self.actualizar_lista_ventanas()
                self.actualizar_estadisticas()
                self.status_var.set(f"🗑️ Ventana {window_id} cerrada")
            except Exception as e:
                self.status_var.set(f"❌ Error cerrando ventana {window_id}: {e}")
                
    def cerrar_todas_ventanas(self):
        """Cerrar todas las ventanas abiertas"""
        if not self.windows:
            messagebox.showinfo("Info", "No hay ventanas abiertas")
            return
            
        if messagebox.askyesno("Confirmar", f"¿Cerrar todas las {len(self.windows)} ventanas?"):
            for window_id in list(self.windows.keys()):
                self.cerrar_ventana(window_id)
            self.status_var.set("🗑️ Todas las ventanas cerradas")
            
    def actualizar_lista_ventanas(self):
        """Actualizar la lista de ventanas en el Treeview"""
        # Limpiar lista actual
        for item in self.windows_tree.get_children():
            self.windows_tree.delete(item)
            
        # Agregar ventanas actuales
        for window_id, info in self.windows.items():
            estado = "✅ Con imagen" if info.get('image_loaded', False) else "⏳ Vacía"
            imagen = info.get('image_name', 'Sin imagen')
            filtros = f"{info.get('filters_count', 0)} filtros"
            
            self.windows_tree.insert('', 'end', values=(
                window_id, estado, imagen, filtros
            ))
            
    def actualizar_estadisticas(self):
        """Actualizar las estadísticas del sistema"""
        ventanas_activas = len(self.windows)
        imagenes_cargadas = sum(1 for info in self.windows.values() if info.get('image_loaded', False))
        
        # Estimación simple de memoria (puedes hacerlo más sofisticado)
        memoria_estimada = ventanas_activas * 50  # MB aproximados por ventana
        
        self.stats_vars['ventanas_activas'].set(str(ventanas_activas))
        self.stats_vars['imagenes_cargadas'].set(str(imagenes_cargadas))
        self.stats_vars['memoria_estimada'].set(f"{memoria_estimada} MB")
        
    def run(self):
        """Iniciar la aplicación"""
        self.root.mainloop()

def main():
    """Función principal"""
    root = tk.Tk()
    app = NASAImageWindowManager(root)
    app.run()

if __name__ == "__main__":
    main()