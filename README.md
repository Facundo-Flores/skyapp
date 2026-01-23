# 🌌 SkyMap — El cielo

**SkyMap** es una aplicación de astronomía hecha con Python que te permite ver qué hay arriba tuyo en tiempo real (o en cualquier momento que elijas). Podés seguir al Sol, la Luna y los planetas con precisión científica, ya sea desde la compu o con el celu en la mano mientras mirás las estrellas.

La app usa **Astropy** para los cálculos de efemérides, así que los datos son posta.

---

## ✨ Qué hace la app

* **Mapa del cielo (2D):** Una proyección polar clásica con Norte arriba. Te marca dónde están los planetas, sus magnitudes (brillo) y si son visibles o están bajo el horizonte.
* **Vista 3D (Experimental):** Un motor hecho en Three.js para que puedas navegar el cielo como si estuvieras en un planetario. Incluye la Vía Láctea posicionada según el Tiempo Sideral Local.
* **Geolocalización:** Si le das permiso, detecta dónde estás para ajustarse automáticamente. Si no, podés cargar las coordenadas a mano.
* **Modo celu:** El diseño se adapta si la abrís desde el celu (achica tipografías y reacomoda los controles para que no sea un lío).
* **Etiquetas Inteligentes:** El algoritmo se encarga de que los nombres de los planetas no se encimen cuando hay conjunciones, así se lee todo bien.

---

## 🛠️ El stack que usé

* **[Streamlit](https://streamlit.io/):** Para armar la interfaz rápido y que sea reactiva.
* **[Astropy](https://www.astropy.org/):** El "cerebro" astronómico. Maneja tiempos, coordenadas y posiciones planetarias.
* **[Matplotlib](https://matplotlib.org/):** Para el renderizado del mapa polar en 2D.
* **[Three.js](https://threejs.org/):** Para la magia de la vista 3D interactiva.
* **BigDataCloud API:** Para el reverse geocoding (pasar de lat/lon a un nombre de ciudad que se entienda).

---

## 🚀 Cómo correrlo en tu máquina

Si querés probarlo localmente, primero cloná el repo:

```bash
git clone https://github.com/Facundo-Flores/skyapp.git
cd skymap
```

Instalá las dependencias (te conviene usar un entorno virtual):

```bash
pip install -r requirements.txt
```

Y después lanzás la app con Streamlit:

```bash
streamlit run app/app_streamlit.py
```


## 📖 Estructura del Proyecto

app_streamlit.py: La cara visible. Maneja los tabs, los inputs y la lógica de Streamlit.

core/sky_core.py: Donde pasa la magia de los cálculos astronómicos y el gráfico 2D.

core/sky_3d.py: Genera el HTML y el JS necesario para el visor WebGL.

assets/: Texturas para que los planetas en 3D no sean simples esferas blancas.


## 🤝 Créditos
Hecho con mucha paciencia y muchos termos de mate. Los datos astronómicos son gracias a la comunidad de Astropy.

Si te gustó, ¡tirale una ⭐ al repo!