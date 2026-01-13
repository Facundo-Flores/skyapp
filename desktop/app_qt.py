from __future__ import annotations

import sys
from typing import Optional

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QMessageBox, QGroupBox,
    QDateTimeEdit, QFileDialog
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.sky_core import compute_altaz, make_figure


# ---------- Worker thread para ubicación Windows (opcional) ----------
class LocationWorker(QThread):
    ok = pyqtSignal(float, float, float)     # lat, lon, alt_m
    err = pyqtSignal(str)

    def run(self):
        try:
            from desktop.location_win import get_windows_location_sync
            res = get_windows_location_sync()
            self.ok.emit(res.latitude, res.longitude, res.altitude_m)
        except Exception as e:
            self.err.emit(str(e))


def fmt_ar_from_time(t_utc: Time) -> str:
    """Devuelve 'dd-mm-aaaa, HH:MM:SS (UTC-3)' a partir de Time UTC."""
    t_art = t_utc - 3 * u.hour
    # t_art.datetime es naive, pero nos sirve para formateo
    return t_art.datetime.strftime("%d-%m-%Y, %H:%M:%S") + " (UTC-3)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mapa del cielo (Desktop) - PyQt + Astropy")
        self.resize(1050, 860)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # --- Grupo ubicación ---
        gb_loc = QGroupBox("Ubicación")
        loc_layout = QHBoxLayout(gb_loc)

        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90.0, 90.0)
        self.lat.setDecimals(6)
        self.lat.setValue(-34.51)
        self.lat.setSingleStep(0.001)

        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180.0, 180.0)
        self.lon.setDecimals(6)
        self.lon.setValue(-58.48)
        self.lon.setSingleStep(0.001)

        self.alt = QDoubleSpinBox()
        self.alt.setRange(-500.0, 9000.0)
        self.alt.setDecimals(1)
        self.alt.setValue(22.0)
        self.alt.setSuffix(" m")

        loc_layout.addWidget(QLabel("Lat:"))
        loc_layout.addWidget(self.lat)
        loc_layout.addWidget(QLabel("Lon:"))
        loc_layout.addWidget(self.lon)
        loc_layout.addWidget(QLabel("Alt:"))
        loc_layout.addWidget(self.alt)

        self.btn_winloc = QPushButton("Usar ubicación de Windows")
        self.btn_winloc.clicked.connect(self.on_windows_location)
        loc_layout.addWidget(self.btn_winloc)

        main.addWidget(gb_loc)

        # --- Grupo fecha/hora (solo display) ---
        gb_time = QGroupBox("Fecha / Hora (Argentina, UTC-3) — solo informativo")
        time_layout = QHBoxLayout(gb_time)

        self.dt_edit = QDateTimeEdit()
        self.dt_edit.setDisplayFormat("dd-MM-yyyy HH:mm:ss")
        self.dt_edit.setCalendarPopup(True)
        self.dt_edit.setReadOnly(True)  # clave: no gobierna el cálculo
        self.dt_edit.setTimeSpec(Qt.TimeSpec.LocalTime)
        self.dt_edit.setDateTime(QDateTime.currentDateTime())

        self.btn_now = QPushButton("Actualizar display a Ahora")
        self.btn_now.clicked.connect(self._sync_display_now)

        time_layout.addWidget(QLabel("Ahora (UTC-3):"))
        time_layout.addWidget(self.dt_edit)
        time_layout.addWidget(self.btn_now)

        main.addWidget(gb_time)

        # --- Acciones ---
        actions = QHBoxLayout()

        self.btn_generate = QPushButton("Generar mapa (AHORA)")
        self.btn_generate.setDefault(True)
        self.btn_generate.clicked.connect(self.on_generate)

        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)

        self.btn_save_png = QPushButton("Guardar PNG…")
        self.btn_save_png.clicked.connect(self.on_save_png)

        self.status = QLabel("Listo.")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        actions.addWidget(self.btn_generate)
        actions.addWidget(self.btn_reset_zoom)
        actions.addWidget(self.btn_save_png)
        actions.addWidget(self.status, stretch=1)

        main.addLayout(actions)

        # --- Canvas Matplotlib ---
        self.canvas: Optional[FigureCanvas] = None
        self._zoom_ax = None
        self._loc_thread: Optional[LocationWorker] = None

        self.canvas_container = QVBoxLayout()
        main.addLayout(self.canvas_container, stretch=1)

        # Primer render
        self.on_generate()

    # ---------- helpers ----------
    def set_status(self, msg: str):
        self.status.setText(msg)

    def _sync_display_now(self):
        # Display: hora local del sistema (en Argentina coincide con UTC-3 si tu PC está en AR)
        self.dt_edit.setDateTime(QDateTime.currentDateTime())

    # ---------- Windows location (opcional) ----------
    def on_windows_location(self):
        self.btn_winloc.setEnabled(False)
        self.set_status("Obteniendo ubicación desde Windows... (revisá permisos)")

        self._loc_thread = LocationWorker()
        self._loc_thread.ok.connect(self._on_loc_ok)
        self._loc_thread.err.connect(self._on_loc_err)
        self._loc_thread.finished.connect(lambda: self.btn_winloc.setEnabled(True))
        self._loc_thread.start()

    def _on_loc_ok(self, lat: float, lon: float, alt_m: float):
        self.lat.setValue(lat)
        self.lon.setValue(lon)
        self.alt.setValue(alt_m if alt_m is not None else 0.0)
        self.set_status(f"Ubicación detectada: lat={lat:.6f}, lon={lon:.6f}, alt={alt_m:.1f} m")

    def _on_loc_err(self, err: str):
        self.set_status("No se pudo obtener ubicación de Windows. Usá ingreso manual.")
        QMessageBox.warning(
            self,
            "Ubicación no disponible",
            "No se pudo obtener la ubicación desde Windows.\n\n"
            "Podés seguir usando lat/lon manual.\n\n"
            f"Detalle: {err}"
        )

    # ---------- Plot / zoom ----------
    def on_generate(self):
        """
        REQUISITO: siempre genera con el horario actual.
        Cálculo en UTC: Time.now()
        Display en UTC-3: formateo.
        """
        try:
            lat = float(self.lat.value())
            lon = float(self.lon.value())
            alt_m = float(self.alt.value())

            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt_m * u.m)

            # Siempre AHORA
            t_utc = Time.now()
            title_ar = fmt_ar_from_time(t_utc)

            # Mantener display sincronizado (solo informativo)
            self._sync_display_now()

            altaz = compute_altaz(t_utc, loc)
            fig = make_figure(altaz, title=f"Cielo visible - {title_ar}")

            self._set_figure(fig)
            self.set_status(f"Mapa generado: {title_ar}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.set_status("Error al generar el mapa.")

    def _set_figure(self, fig):
        # Eliminar canvas anterior si existe
        if self.canvas is not None:
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

        self.canvas = FigureCanvas(fig)
        self.canvas_container.addWidget(self.canvas)

        # Guardar Axes (polar) para zoom
        self._zoom_ax = fig.axes[0] if fig.axes else None

        # Conectar scroll del mouse al zoom
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

    def _on_scroll_zoom(self, event):
        """
        Zoom con rueda del mouse sobre gráfico polar.
        En PolarAxes, el radio usa el eje Y (ylim).
        """
        if self._zoom_ax is None:
            return

        ax = self._zoom_ax

        zoom_step = 0.85
        if event.button == "up":          # scroll adelante
            scale = zoom_step
        elif event.button == "down":      # scroll atrás
            scale = 1.0 / zoom_step
        else:
            return

        rmin, rmax = ax.get_ylim()
        new_rmax = float(rmax) * float(scale)
        new_rmax = max(5.0, min(90.0, new_rmax))  # clamp

        ax.set_ylim(0.0, new_rmax)
        ax.figure.canvas.draw_idle()

    def on_reset_zoom(self):
        if self._zoom_ax is None:
            return
        self._zoom_ax.set_ylim(0.0, 90.0)
        self._zoom_ax.figure.canvas.draw_idle()

    # ---------- Save PNG ----------
    def on_save_png(self):
        if self.canvas is None or self.canvas.figure is None:
            QMessageBox.information(self, "Sin gráfico", "Primero generá un mapa para poder guardarlo.")
            return

        # Nombre sugerido con AHORA (UTC-3)
        t_utc = Time.now()
        t_art = t_utc - 3 * u.hour
        suggested = t_art.datetime.strftime("%Y%m%d_%H%M%S")
        default_name = f"cielo_{suggested}.png"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar PNG",
            default_name,
            "PNG (*.png)"
        )
        if not path:
            return

        try:
            self.canvas.figure.savefig(path, dpi=200)
            self.set_status(f"PNG guardado: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
            self.set_status("Error al guardar PNG.")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
