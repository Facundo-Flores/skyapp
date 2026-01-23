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

        # Proyección astronómica exacta: Norte = -Z, Este = +X
        r = 1000.0
        x = r * np.cos(alt_rad) * np.sin(az_rad)
        y = r * np.sin(alt_rad)
        z = -r * np.cos(alt_rad) * np.cos(az_rad)

        radius = 2.5
        if "Sol" in name:
            radius = 10.0
        elif "Luna" in name:
            radius = 8.0

        pts.append({
            "name": name,
            "x": float(x), "y": float(y), "z": float(z),
            "alt": round(float(c.alt.deg), 2),
            "az": round(float(c.az.deg), 2),
            "texture": tex_map.get(name, ""),
            "radius": radius
        })

    data_json = json.dumps({
        "points": pts,
        "milkyway": tex_map.get("MilkyWay", ""),
        "lst": lst_deg
    })

    return f"""
<!doctype html>
<html>
<head>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }}
        #wrap {{ width: 100%; height: 100%; position: relative; cursor: crosshair; }}

        /* HUD de ubicación (Abajo a la izquierda) */
        #hud-bottom {{ 
            position: absolute; bottom: 20px; left: 20px; 
            color: #00ffcc; font-family: ui-monospace, monospace; 
            font-size: 12px; pointer-events: none; text-shadow: 1px 1px 3px #000;
        }}

        /* Panel de controles (derecha) */
        .ui-panel {{ 
            position: absolute; top: 20px; right: 20px; 
            display: flex; flex-direction: column; gap: 8px; z-index: 1000; 
        }}
        .btn {{ 
            background: rgba(15, 23, 42, 0.9); color: #00ffcc; border: 1px solid #00ffcc; 
            padding: 10px; cursor: pointer; border-radius: 4px; font-family: monospace; font-weight: bold;
            backdrop-filter: blur(4px);
        }}

        /* Etiquetas */
        .label-obj {{ color: #fff; font-family: sans-serif; font-size: 11px; background: rgba(0,0,0,0.7); padding: 2px 5px; border-radius: 3px; pointer-events: none; }}
        .label-cardinal {{ color: #FFD700; font-family: 'Orbitron', sans-serif; font-weight: bold; font-size: 28px; pointer-events: none; }}

        /* Visor central */
        #crosshair {{
            position: absolute; top: 50%; left: 50%;
            width: 30px; height: 30px; border: 1px solid rgba(0, 255, 204, 0.2);
            border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none;
        }}
    </style>
    <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.161.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.161.0/examples/jsm/" }} }}
    </script>
</head>
<body>
    <div id="wrap">
        <div id="crosshair"></div>
        <div class="ui-panel">
            <button class="btn" id="btnIn">+</button>
            <button class="btn" id="btnOut">−</button>
            <button class="btn" id="btnRes">NORTE</button>
        </div>
        <div id="hud-bottom">
            AZ: <span id="vAz">0</span>° | ALT: <span id="vAlt">0</span>°<br>
            <span style="opacity: 0.6">Hacé clic para centrar objeto</span>
        </div>
    </div>

    <script type="module">
        import * as THREE from 'three';
        import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

        const DATA = {data_json};
        const scene = new THREE.Scene();

        let fov = 60;
        const camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, 0.1, 3000);
        camera.position.set(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(1); // Mantiene GPU baja
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        document.getElementById('wrap').appendChild(renderer.domElement);

        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.domElement.style.position = 'absolute';
        labelRenderer.domElement.style.top = '0';
        labelRenderer.domElement.style.pointerEvents = 'none';
        document.getElementById('wrap').appendChild(labelRenderer.domElement);

        const loader = new THREE.TextureLoader();
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        // --- 1. ESTRELLAS Y VÍA LÁCTEA ---
        const starPos = [];
        for(let i=0; i<4000; i++) {{
            const r = 1500;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            starPos.push(r*Math.sin(phi)*Math.cos(theta), r*Math.sin(phi)*Math.sin(theta), r*Math.cos(phi));
        }}
        const starGeo = new THREE.BufferGeometry();
        starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starPos, 3));
        scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({{ color: 0xffffff, size: 1.2 }})));

        if(DATA.milkyway) {{
            loader.load(DATA.milkyway, (tex) => {{
                tex.colorSpace = THREE.SRGBColorSpace;
                const sky = new THREE.Mesh(
                    new THREE.SphereGeometry(1400, 32, 32),
                    new THREE.MeshBasicMaterial({{ map: tex, side: THREE.BackSide, transparent: true, opacity: 0.4 }})
                );
                sky.rotation.y = THREE.MathUtils.degToRad(DATA.lst + 180);
                scene.add(sky);
            }});
        }}

        // --- 2. GRILLA Y PUNTOS CARDINALES ---
        const gridMat = new THREE.LineBasicMaterial({{ color: 0x222222 }});
        for (let a = 0; a <= 90; a += 15) {{
            const rad = 1000 * Math.cos(a * Math.PI / 180);
            const y = 1000 * Math.sin(a * Math.PI / 180);
            const pts = [];
            for (let i = 0; i <= 64; i++) {{
                const th = (i / 64) * Math.PI * 2;
                pts.push(new THREE.Vector3(Math.cos(th)*rad, y, Math.sin(th)*rad));
            }}
            scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
        }}

        const cards = [{{t:'N',az:0}}, {{t:'E',az:90}}, {{t:'S',az:180}}, {{t:'W',az:270}}];
        cards.forEach(c => {{
            const div = document.createElement('div');
            div.className = 'label-cardinal'; div.textContent = c.t;
            const obj = new CSS2DObject(div);
            const r = THREE.MathUtils.degToRad(c.az);
            obj.position.set(Math.sin(r)*980, 0, -Math.cos(r)*980);
            scene.add(obj);
        }});

        // --- 3. PLANETAS (INTERACTIVOS) ---
        const planetMeshes = [];
        DATA.points.forEach(p => {{
            const geo = new THREE.SphereGeometry(p.radius, 16, 16);
            const mat = p.texture ? new THREE.MeshBasicMaterial({{ map: loader.load(p.texture) }}) : new THREE.MeshBasicMaterial({{ color: 0xffffff }});
            if(mat.map) mat.map.colorSpace = THREE.SRGBColorSpace;

            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(p.x, p.y, p.z);
            mesh.userData = {{ az: p.az, alt: p.alt }};
            mesh.lookAt(0,0,0);
            scene.add(mesh);
            planetMeshes.push(mesh);

            const div = document.createElement('div');
            div.className = 'label-obj'; div.textContent = p.name;
            const l = new CSS2DObject(div);
            l.position.set(0, p.radius + 5, 0);
            mesh.add(l);
        }});

        // --- 4. LÓGICA DE NAVEGACIÓN Y CENTRADO ---
        let lon = 0, lat = 0;
        let tLon = 0, tLat = 0, tFov = 60, isAnim = false;

        const syncHUD = () => {{
            document.getElementById('vAz').textContent = Math.round((lon % 360 + 360) % 360);
            document.getElementById('vAlt').textContent = Math.round(lat);
        }};

        // Clic para centrar
        document.getElementById('wrap').addEventListener('click', (e) => {{
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(planetMeshes);
            if (hits.length > 0) {{
                const o = hits[0].object.userData;
                tLon = o.az; tLat = o.alt; tFov = 10; isAnim = true;
            }}
        }});

        // Zoom manual
        document.getElementById('btnIn').onclick = () => {{ fov = Math.max(2, fov - 10); tFov = fov; }};
        document.getElementById('btnOut').onclick = () => {{ fov = Math.min(100, fov + 10); tFov = fov; }};
        document.getElementById('btnRes').onclick = () => {{ tLon = 0; tLat = 0; tFov = 60; isAnim = true; }};

        window.addEventListener('wheel', (e) => {{
            fov = Math.max(2, Math.min(100, fov + e.deltaY * 0.05));
            tFov = fov;
        }}, {{ passive: false }});

        document.addEventListener('pointermove', (e) => {{
            if (e.buttons === 1) {{
                isAnim = false; // Interrumpir animación si el usuario mueve
                lon -= e.movementX * 0.1;
                lat = Math.max(-85, Math.min(85, lat - e.movementY * 0.1));
                syncHUD();
            }}
        }});

        function animate() {{
            requestAnimationFrame(animate);
            if (isAnim) {{
                lon += (tLon - lon) * 0.08;
                lat += (tLat - lat) * 0.08;
                fov += (tFov - fov) * 0.08;
                syncHUD();
                if (Math.abs(lon - tLon) < 0.01) isAnim = false;
            }}
            camera.fov = fov;
            camera.updateProjectionMatrix();

            const phi = THREE.MathUtils.degToRad(90 - lat);
            const theta = THREE.MathUtils.degToRad(lon);
            const target = new THREE.Vector3();
            // CORRECCIÓN DE CENTRADO!!!!:
            target.set(
                Math.sin(theta) * Math.sin(phi),
                Math.cos(phi),
                -Math.cos(theta) * Math.sin(phi)
            );
            camera.lookAt(target);
            renderer.render(scene, camera);
            labelRenderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""