from __future__ import annotations
import sys
import requests
import base64
from pathlib import Path
from typing import Optional

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QMessageBox, QGroupBox,
    QFileDialog, QStatusBar, QCheckBox, QFormLayout, QStackedWidget, QComboBox
)
# Componente para ver el mapa 3D (basado en Chromium)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Importamos tus cores
from core.sky_core import compute_altaz, make_figure
from core.sky_3d import build_sky_3d_html


# --- Utilidad para Texturas ---
def get_local_asset_base64(file_name: str) -> str:
    """Busca la imagen en la carpeta assets y la pasa a Base64 para el HTML 3D."""
    path = Path(__file__).parent.parent / "assets" / file_name
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


# --- Hilo de Cálculo ---
class SkyWorker(QThread):
    finished = pyqtSignal(object, object)  # Envía (Datos/Figura, Tipo)

    def __init__(self, lat, lon, alt, theme, mode_3d=False):
        super().__init__()
        self.lat, self.lon, self.alt, self.theme, self.mode_3d = lat, lon, alt, theme, mode_3d

    def run(self):
        try:
            loc = EarthLocation(lat=self.lat * u.deg, lon=self.lon * u.deg, height=self.alt * u.m)
            t_now = Time.now()
            altaz, _ = compute_altaz(t_now, loc)

            if self.mode_3d:
                file_mapping = {
                    "Sol": "sun.png",
                    "Luna": "moon.png",
                    "Mercurio": "mercury.png",
                    "Venus": "venus.png",
                    "Marte": "mars.png",
                    "Júpiter": "jupiter.png",
                    "Saturno": "saturn.png",
                    "Urano": "uranus.png",
                    "Neptuno": "neptune.png",
                    "MilkyWay": "milkyway.png"
                }
                # Lógica 3D
                tex_map = {k: get_local_asset_base64(v) for k, v in file_mapping.items()}

                lst = t_now.sidereal_time('mean', longitude=loc.lon)
                html = build_sky_3d_html(altaz, tex_map, lst_deg=float(lst.deg))
                self.finished.emit(html, "3d")
            else:
                # Lógica 2D
                fig = make_figure(altaz, title="", theme=self.theme)
                fig.patch.set_facecolor("#121212" if self.theme == "dark" else "#FFFFFF")
                self.finished.emit(fig, "2d")
        except Exception as e:
            self.finished.emit(e, "error")


# --- UI Principal ---
class MainWindow(QMainWindow):
    def __init__(self, theme: str):
        super().__init__()
        self.theme = theme
        self.canvas: Optional[FigureCanvas] = None
        self.init_ui()
        self.run_calculation()

    def init_ui(self):
        self.setWindowTitle("SkyMap Desktop Pro 3D")
        self.resize(1280, 850)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # --- Sidebar ---
        sidebar = QVBoxLayout()

        title = QLabel("🌌 SkyMap")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #2A5BD7; margin-bottom: 10px;")
        sidebar.addWidget(title)

        # Modo de Vista
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["🗺️ Mapa 2D (Estático)", "🌍 Vista 3D (Interactivo)"])
        self.combo_mode.currentIndexChanged.connect(self.run_calculation)
        sidebar.addWidget(QLabel("Modo de visualización:"))
        sidebar.addWidget(self.combo_mode)

        # Ubicación (Inputs rápidos)
        gb_loc = QGroupBox("📍 Ubicación")
        form = QFormLayout(gb_loc)
        self.lat = self._create_sb(-90, 90, -34.51)
        self.lon = self._create_sb(-180, 180, -58.48)
        self.btn_gps = QPushButton("📍 Auto-ubicar")
        self.btn_gps.clicked.connect(self.on_auto_locate)
        form.addRow("Lat:", self.lat)
        form.addRow("Lon:", self.lon)
        form.addRow(self.btn_gps)
        sidebar.addWidget(gb_loc)

        self.btn_theme = QPushButton("🌓 Cambiar Tema")
        self.btn_theme.clicked.connect(self.on_toggle_theme)
        sidebar.addWidget(self.btn_theme)

        sidebar.addStretch()
        layout.addLayout(sidebar, 1)

        # --- Área de Visualización (Stacked) ---
        self.display_stack = QStackedWidget()

        # 1. Contenedor 2D (Matplotlib)
        self.container_2d = QWidget()
        self.layout_2d = QVBoxLayout(self.container_2d)
        self.display_stack.addWidget(self.container_2d)

        # 2. Contenedor 3D (WebEngine)
        self.web_view = QWebEngineView()
        self.display_stack.addWidget(self.web_view)

        layout.addWidget(self.display_stack, 4)

        self.setStatusBar(QStatusBar())

    def _create_sb(self, min_v, max_v, default):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v);
        sb.setValue(default);
        sb.setDecimals(4)
        sb.valueChanged.connect(self.run_calculation)
        return sb

    def run_calculation(self):
        is_3d = (self.combo_mode.currentIndex() == 1)
        self.worker = SkyWorker(self.lat.value(), self.lon.value(), 22.0, self.theme, is_3d)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.statusBar().showMessage("Actualizando...")

    def on_finished(self, result, mode):
        if mode == "error":
            QMessageBox.critical(self, "Error", str(result))
            return

        if mode == "2d":
            self.display_stack.setCurrentIndex(0)
            if self.canvas:
                plt.close(self.canvas.figure)
                self.layout_2d.removeWidget(self.canvas)
            self.canvas = FigureCanvas(result)
            self.layout_2d.addWidget(self.canvas)

        elif mode == "3d":
            self.display_stack.setCurrentIndex(1)
            # Cargamos el HTML generado directamente al browser interno
            self.web_view.setHtml(result)

        self.statusBar().showMessage("Listo.")

    def on_auto_locate(self):
        try:
            r = requests.get("https://ipapi.co/json/").json()
            self.lat.setValue(r['latitude']);
            self.lon.setValue(r['longitude'])
        except:
            pass

    def on_toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        apply_qt_theme(QApplication.instance(), self.theme)
        self.run_calculation()


def apply_qt_theme(app, theme):
    app.setStyle("Fusion")
    if theme == "dark":
        app.setStyleSheet(
            "QMainWindow, QWidget { background: #121212; color: white; } QGroupBox { border: 1px solid #444; }")
    else:
        app.setStyleSheet("QMainWindow, QWidget { background: #F0F0F0; color: black; }")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    apply_qt_theme(app, "dark")
    win = MainWindow("dark")
    win.show()
    sys.exit(app.exec())