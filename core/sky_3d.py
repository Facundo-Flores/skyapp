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


def build_sky_3d_html(altaz_dict, title="Cielo 3D"):
    """
    Versión Optimizada para Orientación:
    - Solo acepta altaz_dict y title para evitar errores de argumentos.
    - Implementa Grilla Altazimutal (líneas de referencia).
    - Navegación tipo 'Survey' (bloqueo de horizonte).
    """
    pts = []
    for name, c in altaz_dict.items():
        alt_rad = np.deg2rad(float(c.alt.deg))
        az_rad = np.deg2rad(float(c.az.deg))

        # Proyección en espacio 3D (Y es arriba):
        # x = cos(alt) * sin(az)  -> Este
        # y = sin(alt)           -> Cénit
        # z = -cos(alt) * cos(az) -> Norte
        r = 100.0
        x = r * np.cos(alt_rad) * np.sin(az_rad)
        y = r * np.sin(alt_rad)
        z = -r * np.cos(alt_rad) * np.cos(az_rad)

        pts.append({
            "name": name,
            "x": float(x), "y": float(y), "z": float(z),
            "color": "#FFFFFF"  # Color base
        })

    data_json = json.dumps({"points": pts, "title": title})

    return f"""
<!doctype html>
<html>
<head>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #05080f; }}
        #wrap {{ width: 100%; height: 100%; position: relative; }}
        .label-cardinal {{ color: #ff5555; font-family: monospace; font-weight: bold; font-size: 26px; }}
        .label-obj {{ color: #00ffcc; font-family: sans-serif; font-size: 11px; background: rgba(0,0,0,0.7); padding: 2px 5px; border-radius: 4px; }}
    </style>
    <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.161.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.161.0/examples/jsm/" }} }}
    </script>
</head>
<body>
    <div id="wrap"></div>
    <script type="module">
        import * as THREE from 'three';
        import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

        const DATA = {data_json};
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 1.6, 0); 

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('wrap').appendChild(renderer.domElement);

        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.domElement.style.position = 'absolute';
        labelRenderer.domElement.style.top = '0';
        document.getElementById('wrap').appendChild(labelRenderer.domElement);

        // --- REFERENCIAS DE ORIENTACIÓN (ESTILO STELLARIUM) ---

        // 1. El Suelo (Evita la sensación de vacío)
        const ground = new THREE.Mesh(
            new THREE.CircleGeometry(150, 64),
            new THREE.MeshBasicMaterial({{ color: 0x080a10, side: THREE.DoubleSide }})
        );
        ground.rotation.x = -Math.PI / 2;
        scene.add(ground);

        // 2. Grilla Altazimutal (Círculos de altura cada 30°)
        const gridMat = new THREE.LineBasicMaterial({{ color: 0x1e293b, transparent: true, opacity: 0.5 }});
        for (let a = 0; a <= 90; a += 30) {{
            const radius = 100 * Math.cos(a * Math.PI / 180);
            const y = 100 * Math.sin(a * Math.PI / 180);
            const points = [];
            for (let i = 0; i <= 64; i++) {{
                const th = (i / 64) * Math.PI * 2;
                points.push(new THREE.Vector3(Math.cos(th) * radius, y, Math.sin(th) * radius));
            }}
            scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), gridMat));
        }}

        // 3. Brújula (N, E, S, O)
        const cardinals = [{{t:'N',az:0}}, {{t:'E',az:90}}, {{t:'S',az:180}}, {{t:'O',az:270}}];
        cardinals.forEach(c => {{
            const div = document.createElement('div');
            div.className = 'label-cardinal'; div.textContent = c.t;
            const obj = new CSS2DObject(div);
            const r = c.az * Math.PI / 180;
            obj.position.set(Math.sin(r)*110, 0, -Math.cos(r)*110);
            scene.add(obj);
        }});

        // --- DIBUJAR OBJETOS ---
        DATA.points.forEach(p => {{
            const mesh = new THREE.Mesh(
                new THREE.SphereGeometry(0.8, 8, 8),
                new THREE.MeshBasicMaterial({{ color: 0xffffff }})
            );
            mesh.position.set(p.x, p.y, p.z);
            scene.add(mesh);

            const div = document.createElement('div');
            div.className = 'label-obj'; div.textContent = p.name;
            const l = new CSS2DObject(div);
            l.position.set(0, 1.5, 0);
            mesh.add(l);
        }});

        // --- NAVEGACIÓN TIPO OBSERVADOR ---
        let lon = 0, lat = 0;
        document.addEventListener('pointermove', (e) => {{
            if (e.buttons === 1) {{
                lon -= e.movementX * 0.15;
                lat = Math.max(-80, Math.min(85, lat - e.movementY * 0.15));
            }}
        }});

        function animate() {{
            requestAnimationFrame(animate);
            const phi = (90 - lat) * Math.PI / 180;
            const theta = lon * Math.PI / 180;
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
