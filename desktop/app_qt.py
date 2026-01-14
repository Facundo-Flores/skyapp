from __future__ import annotations

import sys
from typing import Optional

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from PyQt6.QtCore import Qt, QDateTime, QTimer, QSettings
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QMessageBox, QGroupBox,
    QDateTimeEdit, QFileDialog, QStatusBar, QCheckBox, QFormLayout
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.sky_core import compute_altaz, make_figure


def fmt_ar_from_time(t_utc: Time) -> str:
    t_art = t_utc - 3 * u.hour
    return t_art.datetime.strftime("%d-%m-%Y, %H:%M:%S") + " (UTC-3)"


def detect_system_theme() -> str:
    """
    Intenta detectar theme del sistema.
    1) Qt ColorScheme (si está disponible)
    2) Fallback por luminosidad de la ventana (palette)
    """
    try:
        cs = QGuiApplication.styleHints().colorScheme()
        # Qt.ColorScheme: Dark/Light/Unknown
        if cs == Qt.ColorScheme.Dark:
            return "dark"
        if cs == Qt.ColorScheme.Light:
            return "light"
    except Exception:
        pass

    # Fallback: mirar el color base de la paleta
    pal = QApplication.palette()
    base = pal.window().color()
    # luminancia aproximada
    lum = 0.2126 * base.red() + 0.7152 * base.green() + 0.0722 * base.blue()
    return "dark" if lum < 128 else "light"


def apply_qt_theme(app: QApplication, theme: str) -> None:
    app.setStyle("Fusion")
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    if theme == "dark":
        app.setStyleSheet("""
            QWidget { background: #121212; color: #EAEAEA; font-family: "Segoe UI","Inter","Arial"; }
            QGroupBox { border: 1px solid #2A2A2A; border-radius: 10px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #CFCFCF; }
            QLabel#appTitle { font-size: 18pt; font-weight: 700; padding: 6px; }
            QLabel#subtitle { color: #B8B8B8; padding-bottom: 6px; }
            QPushButton { background: #1E1E1E; border: 1px solid #2E2E2E; border-radius: 10px; padding: 10px 12px; }
            QPushButton:hover { background: #242424; }
            QPushButton:pressed { background: #2A2A2A; }
            QPushButton#primary { background: #2A5BD7; border: 1px solid #2A5BD7; font-weight: 700; }
            QPushButton#primary:hover { background: #2E63E8; }
            QDoubleSpinBox, QDateTimeEdit { background: #1A1A1A; border: 1px solid #2A2A2A; border-radius: 8px; padding: 6px; }
            QCheckBox { padding: 6px; }
            QStatusBar { background: #0F0F0F; border-top: 1px solid #2A2A2A; }
        """)
    else:
        app.setStyleSheet("""
            QWidget { background: #FFFFFF; color: #111111; font-family: "Segoe UI","Inter","Arial"; }
            QGroupBox { border: 1px solid #DDDDDD; border-radius: 10px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #333333; }
            QLabel#appTitle { font-size: 18pt; font-weight: 700; padding: 6px; }
            QLabel#subtitle { color: #555555; padding-bottom: 6px; }
            QPushButton { background: #F4F4F4; border: 1px solid #D0D0D0; border-radius: 10px; padding: 10px 12px; }
            QPushButton:hover { background: #EDEDED; }
            QPushButton:pressed { background: #E3E3E3; }
            QPushButton#primary { background: #2A5BD7; border: 1px solid #2A5BD7; color: #FFFFFF; font-weight: 700; }
            QPushButton#primary:hover { background: #2E63E8; }
            QDoubleSpinBox, QDateTimeEdit { background: #FFFFFF; border: 1px solid #CCCCCC; border-radius: 8px; padding: 6px; }
            QCheckBox { padding: 6px; }
            QStatusBar { background: #F6F6F6; border-top: 1px solid #DDDDDD; }
        """)


class MainWindow(QMainWindow):
    def __init__(self, theme: str):
        super().__init__()
        self.theme = theme  # "dark" / "light"

        self.setWindowTitle("Mapa del cielo (Desktop) - PyQt + Astropy")
        self.resize(1100, 880)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo.")

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        # --- Panel izquierdo ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        title = QLabel("Cielo visible")
        title.setObjectName("appTitle")
        left_panel.addWidget(title)

        subtitle = QLabel("Ubicación y controles")
        subtitle.setObjectName("subtitle")
        left_panel.addWidget(subtitle)

        gb_loc = QGroupBox("Ubicación")
        loc_form = QFormLayout(gb_loc)
        loc_form.setSpacing(8)

        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90.0, 90.0)
        self.lat.setDecimals(6)
        self.lat.setValue(-34.51)

        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180.0, 180.0)
        self.lon.setDecimals(6)
        self.lon.setValue(-58.48)

        self.alt = QDoubleSpinBox()
        self.alt.setRange(-500.0, 9000.0)
        self.alt.setDecimals(1)
        self.alt.setValue(22.0)
        self.alt.setSuffix(" m")

        loc_form.addRow("Latitud:", self.lat)
        loc_form.addRow("Longitud:", self.lon)
        loc_form.addRow("Altitud:", self.alt)
        left_panel.addWidget(gb_loc)

        gb_time = QGroupBox("Fecha / Hora (Argentina, UTC-3) — informativo")
        time_layout = QHBoxLayout(gb_time)

        self.dt_edit = QDateTimeEdit()
        self.dt_edit.setDisplayFormat("dd-MM-yyyy HH:mm:ss")
        self.dt_edit.setCalendarPopup(True)
        self.dt_edit.setReadOnly(True)
        self.dt_edit.setTimeSpec(Qt.TimeSpec.LocalTime)
        self.dt_edit.setDateTime(QDateTime.currentDateTime())

        self.btn_now = QPushButton("Actualizar")
        self.btn_now.clicked.connect(self._sync_display_now)

        time_layout.addWidget(self.dt_edit, stretch=1)
        time_layout.addWidget(self.btn_now)
        left_panel.addWidget(gb_time)

        self.btn_generate = QPushButton("Generar mapa (AHORA)")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.setMinimumHeight(48)
        self.btn_generate.clicked.connect(self.on_generate)

        self.chk_auto_refresh = QCheckBox("Actualizar cada 10s")
        self.chk_auto_refresh.stateChanged.connect(self._toggle_auto_refresh)

        actions_row = QHBoxLayout()
        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)
        self.btn_save_png = QPushButton("Guardar PNG…")
        self.btn_save_png.clicked.connect(self.on_save_png)
        actions_row.addWidget(self.btn_reset_zoom)
        actions_row.addWidget(self.btn_save_png)

        # Toggle de tema (usuario)
        self.btn_toggle_theme = QPushButton()
        self._refresh_theme_button_text()
        self.btn_toggle_theme.clicked.connect(self.on_toggle_theme)

        left_panel.addWidget(self.btn_generate)
        left_panel.addWidget(self.chk_auto_refresh)
        left_panel.addLayout(actions_row)
        left_panel.addWidget(self.btn_toggle_theme)
        left_panel.addStretch(1)

        main.addLayout(left_panel, stretch=0)

        # --- Canvas ---
        self.canvas: Optional[FigureCanvas] = None
        self._zoom_ax = None
        self._auto_timer: Optional[QTimer] = None

        self.canvas_container = QVBoxLayout()
        main.addLayout(self.canvas_container, stretch=1)

        self.on_generate()

    def set_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def _sync_display_now(self):
        self.dt_edit.setDateTime(QDateTime.currentDateTime())

    def _toggle_auto_refresh(self, state: int):
        if state == Qt.CheckState.Checked.value:
            if self._auto_timer is None:
                self._auto_timer = QTimer(self)
                self._auto_timer.timeout.connect(self.on_generate)
            self._auto_timer.start(10_000)
            self.set_status("Auto-refresco activado cada 10s.")
        else:
            if self._auto_timer is not None:
                self._auto_timer.stop()
            self.set_status("Auto-refresco desactivado.")

    def _refresh_theme_button_text(self):
        self.btn_toggle_theme.setText("Tema: Oscuro → Claro" if self.theme == "dark" else "Tema: Claro → Oscuro")

    def on_toggle_theme(self):
        # Cambia tema
        self.theme = "light" if self.theme == "dark" else "dark"
        self._refresh_theme_button_text()

        # Persistir preferencia
        QSettings("skyapp", "skyapp").setValue("theme", self.theme)

        # Aplicar tema a Qt (app global)
        apply_qt_theme(QApplication.instance(), self.theme)

        # Regenerar plot para que también cambie
        self.on_generate()
        self.set_status(f"Tema aplicado: {self.theme}")

    def on_generate(self):
        try:
            lat = float(self.lat.value())
            lon = float(self.lon.value())
            alt_m = float(self.alt.value())
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt_m * u.m)

            t_utc = Time.now()
            title_ar = fmt_ar_from_time(t_utc)

            self._sync_display_now()

            altaz = compute_altaz(t_utc, loc)
            fig = make_figure(altaz, title=f"Cielo visible - {title_ar}", theme=self.theme)

            self._set_figure(fig)
            self.set_status(f"Mapa generado: {title_ar}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.set_status("Error al generar el mapa.")

    def _set_figure(self, fig):
        # Cerrar figura anterior para evitar leaks con autorefrescar activo
        if self.canvas is not None:
            old_fig = self.canvas.figure
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

            if old_fig is not None:
                import matplotlib.pyplot as plt
                plt.close(old_fig)

        self.canvas = FigureCanvas(fig)
        self.canvas_container.addWidget(self.canvas)

        self._zoom_ax = fig.axes[0] if fig.axes else None
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

    def _on_scroll_zoom(self, event):
        if self._zoom_ax is None:
            return
        ax = self._zoom_ax

        zoom_step = 0.85
        if event.button == "up":
            scale = zoom_step
        elif event.button == "down":
            scale = 1.0 / zoom_step
        else:
            return

        _, rmax = ax.get_ylim()
        new_rmax = float(rmax) * float(scale)
        new_rmax = max(5.0, min(90.0, new_rmax))
        ax.set_ylim(0.0, new_rmax)
        ax.figure.canvas.draw_idle()

    def on_reset_zoom(self):
        if self._zoom_ax is None:
            return
        self._zoom_ax.set_ylim(0.0, 90.0)
        self._zoom_ax.figure.canvas.draw_idle()

    def on_save_png(self):
        if self.canvas is None or self.canvas.figure is None:
            QMessageBox.information(self, "Sin gráfico", "Primero generá un mapa para poder guardarlo.")
            return

        t_utc = Time.now()
        t_art = t_utc - 3 * u.hour
        suggested = t_art.datetime.strftime("%Y%m%d_%H%M%S")
        default_name = f"cielo_{suggested}.png"

        path, _ = QFileDialog.getSaveFileName(self, "Guardar PNG", default_name, "PNG (*.png)")
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

    # Preferencia guardada, o tema del sistema si no hay preferencia
    settings = QSettings("skyapp", "skyapp")
    saved = settings.value("theme", None)
    theme = str(saved) if saved in ("dark", "light") else detect_system_theme()

    apply_qt_theme(app, theme)

    w = MainWindow(theme=theme)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
