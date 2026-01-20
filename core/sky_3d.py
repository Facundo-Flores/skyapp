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


def build_sky_3d_html(
    altaz_dict: Dict[str, AltAz],
    title: str = "Cielo 3D (experimental)",
    show_horizon: bool = True,
    show_stars: bool = True,
    stars_n: int = 600,
) -> str:
    """
    Devuelve HTML+JS para embebido en st.components.v1.html.
    Usa Three.js desde CDN (experimental).
    """
    pts = _points_from_altaz(altaz_dict)
    payload = {
        "title": title,
        "points": pts,
        "showHorizon": bool(show_horizon),
        "showStars": bool(show_stars),
        "starsN": int(stars_n),
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {{
      margin: 0; padding: 0;
      height: 100%;
      background: transparent;
      overflow: hidden;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Roboto Mono", monospace;
    }}
    #wrap {{
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 16px;
      overflow: hidden;
      background: radial-gradient(circle at 30% 20%, rgba(99,102,241,0.12), rgba(10,14,26,0.0) 55%),
                  linear-gradient(180deg, rgba(10,14,26,0.0), rgba(10,14,26,0.35));
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .hud {{
      position: absolute;
      left: 12px;
      top: 10px;
      right: 12px;
      z-index: 10;
      pointer-events: none;
      color: rgba(226,232,240,0.92);
      text-shadow: 0 2px 14px rgba(0,0,0,0.5);
      font-size: 14px;
      letter-spacing: 0.2px;
    }}
    .hud .sub {{
      margin-top: 4px;
      font-size: 12px;
      color: rgba(148,163,184,0.9);
    }}
    .label {{
      color: rgba(226,232,240,0.95);
      font-size: 12px;
      padding: 2px 6px;
      border-radius: 999px;
      background: rgba(2,6,23,0.55);
      border: 1px solid rgba(148,163,184,0.22);
      backdrop-filter: blur(4px);
      transform: translate(-50%, -125%);
      white-space: nowrap;
      user-select: none;
      pointer-events: none;
    }}
    .hint {{
      position: absolute;
      right: 10px;
      bottom: 10px;
      z-index: 10;
      font-size: 12px;
      color: rgba(148,163,184,0.9);
      background: rgba(15,17,21,0.25);
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 12px;
      padding: 6px 10px;
      pointer-events: none;
    }}
    #btnCenter {{
      position: absolute;
      right: 10px;
      top: 10px;
      z-index: 12;
      background: rgba(2,6,23,0.55);
      color: rgba(226,232,240,0.92);
      border: 1px solid rgba(148,163,184,0.22);
      padding: 6px 10px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
    }}
    #btnCenter:active {{
      transform: translateY(1px);
    }}
  </style>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://unpkg.com/three@0.161.0/build/three.module.js",
      "three/addons/controls/OrbitControls.js": "https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js",
      "three/addons/renderers/CSS2DRenderer.js": "https://unpkg.com/three@0.161.0/examples/jsm/renderers/CSS2DRenderer.js"
    }}
  }}
  </script>
</head>

<body>
  <div id="wrap">
    <div class="hud">
      <div id="title"></div>
      <div class="sub">Arrastrá para rotar • Ruedita/gesto para zoom • Doble click: centrar • Click: enfocar</div>
    </div>
    <button id="btnCenter">Centrar</button>
    <div class="hint">Vista 3D experimental</div>
  </div>

  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ CSS2DRenderer, CSS2DObject }} from "three/addons/renderers/CSS2DRenderer.js";

    const DATA = {data_json};

    const wrap = document.getElementById("wrap");
    const titleEl = document.getElementById("title");
    titleEl.textContent = DATA.title || "Cielo 3D (experimental)";

    // Scene
    const scene = new THREE.Scene();

    // Camera (arranca en "nivel suelo mirando al norte")
    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 80);

    // Renderer (WebGL)
    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    wrap.appendChild(renderer.domElement);

    // Renderer de labels (DOM)
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(wrap.clientWidth, wrap.clientHeight);
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.top = "0";
    labelRenderer.domElement.style.left = "0";
    labelRenderer.domElement.style.pointerEvents = "none";
    wrap.appendChild(labelRenderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.45;
    controls.maxDistance = 5.5;

    // --- “Preset” suelo mirando al norte ---
    function setGroundNorthView() {{
      const eyeZ = 0.06;
      camera.position.set(0.0, 0.0, eyeZ);
      controls.target.set(0.0, 1.0, eyeZ);
      controls.update();
    }}

    // Luz
    scene.add(new THREE.AmbientLight(0xffffff, 0.78));
    const dir = new THREE.DirectionalLight(0xffffff, 0.75);
    dir.position.set(2.2, -0.2, 2.4);
    scene.add(dir);

    // Mundo (para encuadrar)
    const world = new THREE.Group();
    scene.add(world);

    // Suelo / grid (más chico y MUCHO más sutil para que no "corte" la esfera)
    const grid = new THREE.GridHelper(2.4, 12, 0x4f46e5, 0x94a3b8);
    grid.material.opacity = 0.07;
    grid.material.transparent = true;
    grid.position.set(0, 0, 0);
    world.add(grid);

    // Horizonte (círculo)
    if (DATA.showHorizon) {{
      const horizonMat = new THREE.LineBasicMaterial({{ color: 0xaab2c3, transparent: true, opacity: 0.28 }});
      const N = 300;
      const pts = [];
      for (let i=0; i<=N; i++) {{
        const th = (i / N) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.sin(th), Math.cos(th), 0));
      }}
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(geo, horizonMat);
      world.add(line);
    }}

    // Domo (hemisferio “cielo” wireframe, más presente)
    {{
      const domeGeo = new THREE.SphereGeometry(1.0, 36, 20, 0, Math.PI*2, 0, Math.PI/2);
      const domeMat = new THREE.MeshBasicMaterial({{
        color: 0x8be9fd,
        wireframe: true,
        transparent: true,
        opacity: 0.085
      }});
      const dome = new THREE.Mesh(domeGeo, domeMat);
      world.add(dome);
    }}

    // Cardinales
    function addCardinal(txt, x, y) {{
      const div = document.createElement("div");
      div.className = "label";
      div.style.background = "rgba(15,17,21,0.18)";
      div.style.border = "1px solid rgba(148,163,184,0.14)";
      div.textContent = txt;
      const obj = new CSS2DObject(div);
      obj.position.set(x, y, 0.02);
      world.add(obj);
    }}
    addCardinal("N", 0, 1.08);
    addCardinal("E", 1.08, 0);
    addCardinal("S", 0, -1.08);
    addCardinal("O", -1.08, 0);

    // --- Estrellas ---
    // 1) Fondo: esfera grande alrededor (para que NO parezca “cortado”)
    function addBackgroundStars(n=1200, radius=18) {{
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(n*3);

      for (let i=0;i<n;i++) {{
        const u = Math.random();
        const v = Math.random();
        const theta = 2*Math.PI*u;
        const phi = Math.acos(2*v - 1);
        const r = radius*(0.9 + 0.1*Math.random());

        pos[3*i+0] = r*Math.sin(phi)*Math.cos(theta);
        pos[3*i+1] = r*Math.cos(phi);
        pos[3*i+2] = r*Math.sin(phi)*Math.sin(theta);
      }}

      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const mat = new THREE.PointsMaterial({{
        color: 0xffffff,
        size: 0.055,
        transparent: true,
        opacity: 0.35,
        sizeAttenuation: true
      }});
      const pts = new THREE.Points(geo, mat);
      scene.add(pts); // afuera del world
    }}

    // 2) Hemisferio: estrellas “sobre el domo” (cielo visible)
    function addHemisphereStars(n=700) {{
      const positions = new Float32Array(n * 3);
      for (let i=0; i<n; i++) {{
        const u = Math.random();
        const v = Math.random();

        const az = 2 * Math.PI * u;
        const k = 0.55; // sesgo hacia horizonte
        const altRad = Math.asin(Math.pow(v, k));

        const x = Math.cos(altRad) * Math.sin(az);
        const y = Math.cos(altRad) * Math.cos(az);
        const z = Math.sin(altRad);

        positions[i*3+0] = x;
        positions[i*3+1] = y;
        positions[i*3+2] = z;
      }}

      const starsGeo = new THREE.BufferGeometry();
      starsGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

      const starsMat = new THREE.PointsMaterial({{
        color: 0xffffff,
        size: 0.016,
        transparent: true,
        opacity: 0.50,
        sizeAttenuation: true
      }});

      const stars = new THREE.Points(starsGeo, starsMat);
      world.add(stars);
    }}

    if (DATA.showStars) {{
      const n = Math.max(200, Math.min(4000, DATA.starsN || 900));
      addBackgroundStars(Math.floor(n*1.3), 18);
      addHemisphereStars(n);
    }}

    // --- Objetos (planetas, luna, etc.) ---
    const objectsGroup = new THREE.Group();
    world.add(objectsGroup);

    function hexToInt(hex) {{
      return parseInt(String(hex || "#ffffff").replace("#",""), 16);
    }}

    // Halo (sprite) para que “salten”
    function makeHalo(colorInt) {{
      const canvas = document.createElement("canvas");
      canvas.width = 128; canvas.height = 128;
      const ctx = canvas.getContext("2d");

      const grd = ctx.createRadialGradient(64, 64, 6, 64, 64, 64);
      grd.addColorStop(0.0, "rgba(255,255,255,0.95)");
      grd.addColorStop(0.25, "rgba(255,255,255,0.35)");
      grd.addColorStop(1.0, "rgba(255,255,255,0.0)");
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, 128, 128);

      const tex = new THREE.CanvasTexture(canvas);
      const mat = new THREE.SpriteMaterial({{
        map: tex,
        color: colorInt,
        transparent: true,
        opacity: 0.55,
        depthWrite: false
      }});
      const spr = new THREE.Sprite(mat);
      return spr;
    }}

    // Raycaster
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const clickable = [];

    for (const p of (DATA.points || [])) {{
      const colorInt = hexToInt(p.color || "#ffffff");

      // Tamaño visible: factor con clamp (en unidades del mundo)
      // Notá: domo radio = 1, entonces 0.03..0.12 se ven bien
      const base = (p.size || 10);
      const radius = THREE.MathUtils.clamp(0.030 + (base / 220.0), 0.035, 0.12);

      const geom = new THREE.SphereGeometry(radius, 20, 16);

      const mat = new THREE.MeshStandardMaterial({{
        color: colorInt,
        roughness: 0.55,
        metalness: 0.05,
        emissive: colorInt,
        emissiveIntensity: 0.18
      }});

      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(p.x, p.y, p.z);
      mesh.userData = p;
      objectsGroup.add(mesh);
      clickable.push(mesh);

      // Halo
      const halo = makeHalo(colorInt);
      halo.scale.set(radius*6.0, radius*6.0, 1);
      halo.position.set(0,0,0);
      mesh.add(halo);

      // Label
      const div = document.createElement("div");
      div.className = "label";
      div.textContent = p.name;

      const label = new CSS2DObject(div);
      label.position.set(0, 0, radius*2.2);
      mesh.add(label);
    }}

    // Encadre (centrar) — incluye el domo para evitar “cortar cuadrantes”
    function frameWorld(fitOffset=1.35) {{
      const box = new THREE.Box3().setFromObject(world);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());

      const maxSize = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      let dist = Math.abs((maxSize / 2) / Math.tan(fov / 2));
      dist *= fitOffset;

      // Ponemos la cámara en una posición “natural” detrás del Sur mirando al Norte,
      // pero a distancia suficiente como para encuadrar el mundo completo.
      const dir = new THREE.Vector3(0, -1, 0.12).normalize(); // desde Sur y un poco arriba
      camera.position.copy(center).add(dir.multiplyScalar(dist));

      camera.near = Math.max(0.01, maxSize / 200);
      camera.far = maxSize * 200;
      camera.updateProjectionMatrix();

      controls.target.copy(center);
      controls.update();
    }}

    // Click: enfocar objeto
    renderer.domElement.addEventListener("click", (event) => {{
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);

      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(clickable, false);
      if (hits.length) {{
        const obj = hits[0].object;
        controls.target.lerp(obj.position, 0.65);
      }}
    }});

    // Doble click: centrar
    renderer.domElement.addEventListener("dblclick", () => setGroundNorthView());

    // Botón centrar
    document.getElementById("btnCenter").addEventListener("click", () => setGroundNorthView());

    // Resize
    function onResize() {{
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      labelRenderer.setSize(w, h);
    }}
    window.addEventListener("resize", onResize);

    // Init: 1) preset “suelo mirando al norte”, 2) frame suave (por si el domo queda cortado)
    onResize();
    setGroundNorthView();

    // Loop
    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
""".strip()
