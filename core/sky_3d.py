# sky_3d.py

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from astropy.coordinates import AltAz

from core.sky_core import MAGS, PLANET_COLORS  # reutilizamos constantes


def _points_from_altaz(altaz_dict: Dict[str, AltAz]) -> List[dict]:
    """
    Convierte AltAz -> puntos 3D sobre un hemisferio de radio 1.
    Convención:
      - Az=0 Norte, 90 Este
      - Alt=0 horizonte, 90 zénit
    Ejes:
      - +Y = Norte
      - +X = Este
      - +Z = Arriba
    """
    pts: List[dict] = []

    for name, c in altaz_dict.items():
        alt = float(c.alt.deg)
        if alt <= 0:
            continue
        az = float(c.az.deg)

        alt_rad = np.deg2rad(alt)
        az_rad = np.deg2rad(az)

        r = 1.0
        x = r * np.cos(alt_rad) * np.sin(az_rad)  # Este
        y = r * np.cos(alt_rad) * np.cos(az_rad)  # Norte
        z = r * np.sin(alt_rad)                   # Arriba

        mag = float(MAGS.get(name, 1.0))

        # Tamaño base (brillante = más grande). Ajustado para que “se vea” en 3D.
        # mag menor => más grande
        size = float(np.clip(22 - mag * 2.8, 6.0, 36.0))

        pts.append(
            {
                "name": name,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "mag": mag,
                "size": size,
                "color": PLANET_COLORS.get(name, "#FFFFFF"),
                "alt": alt,
                "az": az,
            }
        )

    return pts


def build_sky_3d_html(altaz_dict, tex_map, lst_deg=0):
    pts = []
    for name, c in altaz_dict.items():
        alt_rad = np.deg2rad(float(c.alt.deg))
        az_rad = np.deg2rad(float(c.az.deg))
        r = 120.0  # Radio del domo un poco más grande
        x = r * np.cos(alt_rad) * np.sin(az_rad)
        y = r * np.sin(alt_rad)
        z = -r * np.cos(alt_rad) * np.cos(az_rad)

        pts.append({
            "name": name, "x": float(x), "y": float(y), "z": float(z),
            "alt": round(float(c.alt.deg), 1),
            "texture": tex_map.get(name, ""),
            "radius": 6 if "Sol" in name else (4 if "Luna" in name else 1.8)
        })

    data_json = json.dumps({"points": pts, "milkyway": tex_map.get("MilkyWay", ""), "lst": lst_deg})

    return f"""
<!doctype html>
<html>
<head>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }}
        #wrap {{ width: 100%; height: 100%; position: relative; }}
        .ui {{ position: absolute; top: 10px; right: 10px; z-index: 100; }}
        .btn {{ background: rgba(0,0,0,0.7); color: white; border: 1px solid #555; padding: 8px; cursor: pointer; border-radius: 4px; }}
        .label-obj {{ color: #00ffcc; font-family: sans-serif; font-size: 11px; text-shadow: 1px 1px 2px #000; pointer-events: none; }}
    </style>
    <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.161.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.161.0/examples/jsm/" }} }}
    </script>
</head>
<body>
    <div id="wrap">
        <div class="ui"><button class="btn" id="res">Reiniciar Vista</button></div>
    </div>

    <script type="module">
        import * as THREE from 'three';
        import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

        const DATA = {data_json};
        const scene = new THREE.Scene();

        const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 2000);
        camera.position.set(0, 1.6, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: false, powerPreference: "high-performance" }});
        renderer.setSize(window.innerWidth, window.innerHeight);

        // --- ARREGLO DE COLOR Y GPU ---
        renderer.setPixelRatio(1); // Forzar 1 para bajar uso de GPU
        renderer.outputColorSpace = THREE.SRGBColorSpace; // Arregla texturas blancas/lavadas

        document.getElementById('wrap').appendChild(renderer.domElement);

        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.domElement.style.position = 'absolute';
        labelRenderer.domElement.style.top = '0';
        document.getElementById('wrap').appendChild(labelRenderer.domElement);

        const loader = new THREE.TextureLoader();

        // --- FONDO ESTELAR MATEMÁTICO (Sin imagen pesada) ---
        const starPos = [];
        for(let i=0; i<2500; i++) {{
            const r = 600;
            const theta = 2 * Math.PI * Math.random();
            const phi = Math.acos(2 * Math.random() - 1);
            starPos.push(r*Math.sin(phi)*Math.cos(theta), Math.abs(r*Math.sin(phi)*Math.sin(theta)), r*Math.cos(phi));
        }}
        const starGeo = new THREE.BufferGeometry();
        starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
        scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({{ color: 0xffffff, size: 0.7 }})));

        // --- CARGAR VÍA LÁCTEA ---
        if(DATA.milkyway) {{
            loader.load(DATA.milkyway, (t) => {{
                t.colorSpace = THREE.SRGBColorSpace;
                const sky = new THREE.Mesh(
                    new THREE.SphereGeometry(700, 32, 32),
                    new THREE.MeshBasicMaterial({{ map: t, side: THREE.BackSide, transparent: true, opacity: 0.4 }})
                );
                sky.rotation.y = THREE.MathUtils.degToRad(DATA.lst);
                scene.add(sky);
            }});
        }}

        // --- OBJETOS CON TEXTURA ---
        DATA.points.forEach(p => {{
            // Geometría reducida (16x16) para ahorrar GPU
            const geo = new THREE.SphereGeometry(p.radius, 16, 16);

            if(p.texture) {{
                loader.load(p.texture, (t) => {{
                    t.colorSpace = THREE.SRGBColorSpace; // IMPORTANTE: Corrige el "blanco"
                    const mat = new THREE.MeshBasicMaterial({{ map: t, transparent: true }});
                    const mesh = new THREE.Mesh(geo, mat);
                    mesh.position.set(p.x, p.y, p.z);
                    mesh.rotation.y = Math.PI;
                    scene.add(mesh);

                    const div = document.createElement('div');
                    div.className = 'label-obj'; div.innerHTML = `<b>${{p.name}}</b><br>${{p.alt}}°`;
                    const l = new CSS2DObject(div);
                    l.position.set(0, p.radius + 2, 0);
                    mesh.add(l);
                }});
            }}
        }});

        // --- NAVEGACIÓN ---
        let lon = 0, lat = 10;
        document.getElementById('res').onclick = () => {{ lon = 0; lat = 10; }};
        document.addEventListener('pointermove', (e) => {{
            if (e.buttons === 1) {{
                lon -= e.movementX * 0.15;
                lat = Math.max(-85, Math.min(85, lat - e.movementY * 0.15));
            }}
        }});

        function animate() {{
            requestAnimationFrame(animate);
            const phi = THREE.MathUtils.degToRad(90 - lat);
            const theta = THREE.MathUtils.degToRad(lon);
            const target = new THREE.Vector3();
            target.setFromSphericalCoords(1, phi, theta);
            camera.lookAt(camera.position.clone().add(target));
            renderer.render(scene, camera);
            labelRenderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""