// 3D globe: physically-lit Earth (real sun for the simulated date, city lights at night, ocean glint, relief),
// scattering atmosphere, INDRA's live fields, lit clouds with shadows, wind streaks, falling rain, lightning.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { mesh } from 'topojson-client';
import { BBOX, TEX_W, TEX_H, CLOUD_W, CLOUD_H, NLAT, NLON, LAT0, LON0, CELLS, bilinear } from './field.js';

const D2R = Math.PI / 180;
export function ll2v(lat, lon, r = 1, out = new THREE.Vector3()) {
  const phi = (lon + 180) * D2R, th = (90 - lat) * D2R;
  return out.set(-Math.cos(phi) * Math.sin(th) * r, Math.cos(th) * r, Math.sin(phi) * Math.sin(th) * r);
}
function v2ll(v) {
  const r = v.length(), lat = 90 - Math.acos(v.y / r) / D2R;
  let lon = Math.atan2(v.z, -v.x) / D2R - 180; if (lon < -180) lon += 360;
  return [lat, lon];
}
/** sub-solar point at 06:30 UTC (local noon over central India) on a given date */
export function sunLatLon(date) {
  const doy = (Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) - Date.UTC(date.getFullYear(), 0, 0)) / 864e5;
  return [-23.44 * Math.cos(2 * Math.PI / 365 * (doy + 10)), 82.5];
}

const VERT = `varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main(){ vUv=uv; vN=normalize(mat3(modelMatrix)*normal); vec4 w=modelMatrix*vec4(position,1.); vP=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`;
const EARTH_FRAG = `uniform sampler2D dayMap, nightMap, waterMap, bumpMap; uniform vec3 sunDir; uniform float light;
varying vec2 vUv; varying vec3 vN; varying vec3 vP;
void main(){
  vec3 N=normalize(vN);
  // relief: perturb the normal with the topography gradient (screen-space derivatives)
  float h=texture2D(bumpMap,vUv).r; vec3 dx=dFdx(vP), dy=dFdy(vP); float hx=dFdx(h), hy=dFdy(h);
  vec3 r1=cross(dy,N), r2=cross(N,dx); float det=dot(dx,r1);
  N=normalize(abs(det)*N - 0.012*sign(det)*(hx*r1+hy*r2));
  float d=dot(N,sunDir), day=smoothstep(-0.10,0.22,d);
  vec3 V=normalize(cameraPosition-vP), H=normalize(sunDir+V);
  vec3 dayC=texture2D(dayMap,vUv).rgb, nightC=texture2D(nightMap,vUv).rgb;
  float water=texture2D(waterMap,vUv).r;
  vec3 lit=dayC*(0.05+1.15*max(d,0.)) + dayC*light*0.35;
  vec3 col=mix(nightC*vec3(1.,.82,.55)*1.6*(1.-0.7*water) + dayC*(0.025+light*0.3), lit, day);
  col+=vec3(1.,.93,.8)*pow(max(dot(N,H),0.),70.)*water*0.9*day;                     // sun glint on oceans
  float f=pow(1.-max(dot(normalize(vN),V),0.),3.);
  col+=vec3(.30,.55,1.)*f*(0.15+0.75*day);                                          // atmospheric limb haze
  gl_FragColor=vec4(col,1.);
  #include <colorspace_fragment>
}`;
const ATMO_FRAG = `uniform vec3 sunDir; uniform float light; varying vec3 vN; varying vec3 vP; varying vec2 vUv;
void main(){ vec3 V=normalize(cameraPosition-vP); float rim=pow(max(0.,.55-dot(normalize(vN),V)),2.2)*3.;
  float sun=smoothstep(-.35,.5,dot(normalize(vN),sunDir));
  vec3 c=mix(vec3(.05,.08,.25),vec3(.35,.62,1.),sun)+vec3(1.,.5,.25)*pow(1.-abs(dot(normalize(vN),sunDir)),8.)*.35;
  gl_FragColor=vec4(c*rim*(0.35+0.65*sun)*(1.-light*0.35),1.); }`;

function glowTexture() {
  const c = Object.assign(document.createElement('canvas'), { width: 64, height: 64 }), g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(190,210,255,.8)'); gr.addColorStop(1, 'rgba(120,150,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c);
}

export class Globe {
  constructor(canvas, { onHover, onPick }) {
    this.canvas = canvas; this.onHover = onHover; this.onPick = onPick; this.active = true; this.autoRotate = false;
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.copy(ll2v(10, -20, 4.2));
    const c = this.controls = new OrbitControls(this.camera, canvas);
    c.enableDamping = true; c.dampingFactor = 0.07; c.minDistance = 1.18; c.maxDistance = 6; c.rotateSpeed = 0.45; c.enablePan = false; c.zoomSpeed = 0.7;

    const loader = new THREE.TextureLoader();
    const tex = (u, srgb = true) => { const t = loader.load(u); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };
    this.ready = new Promise(res => { THREE.DefaultLoadingManager.onLoad = res; });
    this.sunDir = new THREE.Vector3(1, 0, 0);

    this.sky = new THREE.Mesh(new THREE.SphereGeometry(40, 32, 32), new THREE.MeshBasicMaterial({ map: tex('/tex/night-sky.png'), side: THREE.BackSide, color: 0x9aa8c8 }));
    this.scene.add(this.sky);
    this.uni = { dayMap: { value: tex('/tex/earth-blue-marble.jpg') }, nightMap: { value: tex('/tex/earth-night.jpg') },
      waterMap: { value: tex('/tex/earth-water.png', false) }, bumpMap: { value: tex('/tex/earth-topology.png', false) },
      sunDir: { value: this.sunDir }, light: { value: 0 } };
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(1, 160, 120), new THREE.ShaderMaterial({ uniforms: this.uni, vertexShader: VERT, fragmentShader: EARTH_FRAG }));
    this.scene.add(this.earth);
    this.scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.075, 96, 64), new THREE.ShaderMaterial({
      uniforms: this.uni, vertexShader: VERT, fragmentShader: ATMO_FRAG, side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending })));

    this.ambient = new THREE.AmbientLight(0xb8c8e8, 0.35); this.scene.add(this.ambient);
    this.sun = new THREE.DirectionalLight(0xfff4e6, 2.6); this.scene.add(this.sun);

    // INDRA's fields and clouds live on patches over the simulated domain; they are lit by the same sun
    const patch = (r, canvas, mat) => {
      const t = new THREE.CanvasTexture(canvas); t.colorSpace = THREE.SRGBColorSpace; t.minFilter = THREE.LinearFilter;
      const g = new THREE.SphereGeometry(r, 96, 96, (BBOX.w + 180) * D2R, (BBOX.e - BBOX.w) * D2R, (90 - BBOX.n) * D2R, (BBOX.n - BBOX.s) * D2R);
      const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ map: t, transparent: true, depthWrite: false, ...mat }));
      this.scene.add(m); return { tex: t, mesh: m };
    };
    this.fieldCanvas = Object.assign(document.createElement('canvas'), { width: TEX_W, height: TEX_H });
    this.cloudCanvas = Object.assign(document.createElement('canvas'), { width: CLOUD_W, height: CLOUD_H });
    this.field = patch(1.0012, this.fieldCanvas, { opacity: 0.85, emissive: 0x222222 });
    this.shadow = patch(1.004, this.cloudCanvas, { color: 0x000000, opacity: 0.32 });
    this.clouds = patch(1.014, this.cloudCanvas, { opacity: 1 });
    this.shadow.tex = this.clouds.tex; this.shadow.mesh.material.map = this.clouds.tex;
    this.shadow.mesh.rotation.y = -0.006;                     // offset so clouds cast a visible shadow

    // wind streaks
    this.N = 2600; this.parts = new Float32Array(this.N * 3); this.age = new Float32Array(this.N);
    for (let k = 0; k < this.N; k++) this.respawn(k, true);
    this.windGeo = new THREE.BufferGeometry();
    this.windPos = new Float32Array(this.N * 6); this.windCol = new Float32Array(this.N * 6);
    this.windGeo.setAttribute('position', new THREE.BufferAttribute(this.windPos, 3));
    this.windGeo.setAttribute('color', new THREE.BufferAttribute(this.windCol, 3));
    this.wind = new THREE.LineSegments(this.windGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scene.add(this.wind);

    // falling rain: short radial streaks spawned where INDRA says it rains, density ~ intensity
    this.RN = 4000; this.rain = new Float32Array(this.RN * 4);  // lat, lon, height, speed
    this.rainGeo = new THREE.BufferGeometry(); this.rainPos = new Float32Array(this.RN * 6); this.rainCol = new Float32Array(this.RN * 6);
    this.rainGeo.setAttribute('position', new THREE.BufferAttribute(this.rainPos, 3));
    this.rainGeo.setAttribute('color', new THREE.BufferAttribute(this.rainCol, 3));
    this.rainMesh = new THREE.LineSegments(this.rainGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.scene.add(this.rainMesh);

    // lightning: a pool of glowing sprites in heavy-rain cells
    const glow = glowTexture();
    this.bolts = Array.from({ length: 8 }, () => { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0 })); s.scale.setScalar(0.05); this.scene.add(s); return { s, life: 0 }; });

    // domain outline + shock marker
    const box = [];
    for (let lo = BBOX.w; lo <= BBOX.e; lo += 1) box.push(ll2v(BBOX.n, lo, 1.005));
    for (let la = BBOX.n; la >= BBOX.s; la -= 1) box.push(ll2v(la, BBOX.e, 1.005));
    for (let lo = BBOX.e; lo >= BBOX.w; lo -= 1) box.push(ll2v(BBOX.s, lo, 1.005));
    for (let la = BBOX.s; la <= BBOX.n; la += 1) box.push(ll2v(la, BBOX.w, 1.005));
    this.domain = new THREE.Line(new THREE.BufferGeometry().setFromPoints(box), new THREE.LineDashedMaterial({ color: 0x46d3ff, transparent: true, opacity: 0.5, dashSize: 0.01, gapSize: 0.008 })).computeLineDistances();
    this.scene.add(this.domain);
    this.shock = new THREE.Group(); this.scene.add(this.shock);
    this.shockRing = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff7a45, transparent: true }));
    this.shockPulse = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff3d77, transparent: true }));
    this.shock.add(this.shockRing, this.shockPulse);

    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2(); this.t0 = performance.now();
    canvas.addEventListener('pointermove', e => this.pointer(e, false));
    canvas.addEventListener('click', e => { if (e.shiftKey) this.pointer(e, true); });
    canvas.addEventListener('pointerleave', () => this.onHover(null));
    canvas.addEventListener('pointerdown', () => { this.intro = null; });
    addEventListener('resize', () => this.resize()); this.resize();
    this.setDate(new Date(2021, 5, 1));
    this.intro = { t: 0, from: this.camera.position.clone(), to: ll2v(20, 82, 2.55) };
  }

  async loadBorders() {
    const topo = await (await fetch('/countries-50m.json')).json();
    const m = mesh(topo, topo.objects.countries), pts = [];
    for (const line of m.coordinates) for (let k = 1; k < line.length; k++) pts.push(ll2v(line[k - 1][1], line[k - 1][0], 1.0022), ll2v(line[k][1], line[k][0], 1.0022));
    this.borders = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xe6f2ff, transparent: true, opacity: 0.42 }));
    this.scene.add(this.borders);
  }

  setDate(date) {
    const [la, lo] = sunLatLon(date); ll2v(la, lo, 1, this.sunDir).normalize();
    this.sun.position.copy(this.sunDir).multiplyScalar(10);
  }
  setTheme(theme) {
    const light = theme === 'light';
    this.uni.light.value = light ? 1 : 0;
    this.sky.visible = !light;
    this.scene.background = light ? new THREE.Color(0xdfe8f3) : null;
    this.ambient.intensity = light ? 0.9 : 0.35;
    if (this.borders) this.borders.material.color.set(light ? 0x1b2a40 : 0xe6f2ff);
  }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  pointer(e, pick) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.set((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1);
    this.ray.setFromCamera(this.mouse, this.camera);
    const hit = this.ray.intersectObject(this.earth)[0];
    if (!hit) return this.onHover(null);
    const [lat, lon] = v2ll(hit.point);
    if (pick) this.onPick(lat, lon); else this.onHover({ lat, lon, x: e.clientX, y: e.clientY });
  }

  /** current simulated day -> wind particles, rain spawn distribution, lightning cells */
  setDay(day) {
    this.u = day ? day.subarray(2 * CELLS, 3 * CELLS) : null; this.v = day ? day.subarray(3 * CELLS, 4 * CELLS) : null;
    if (!day) return;
    const tp = day.subarray(CELLS, 2 * CELLS), cdf = new Float32Array(CELLS); let s = 0; this.storm = [];
    for (let k = 0; k < CELLS; k++) { const r = tp[k] > 2 ? Math.pow(tp[k], 1.3) : 0; s += r; cdf[k] = s; if (tp[k] > 35) this.storm.push(k); }
    this.rainCdf = cdf; this.rainTot = s;
  }
  spawnRain(k) {
    const R = this.rain; if (!this.rainTot) { R[k * 4 + 2] = -1; return; }
    const x = Math.random() * this.rainTot; let lo = 0, hi = CELLS - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.rainCdf[m] < x) lo = m + 1; else hi = m; }
    R[k * 4] = LAT0 + Math.floor(lo / NLON) + Math.random() - 0.5; R[k * 4 + 1] = LON0 + lo % NLON + Math.random() - 0.5;
    R[k * 4 + 2] = Math.random(); R[k * 4 + 3] = 0.8 + Math.random() * 0.6;
  }
  respawn(k, randomAge) {
    this.parts[k * 3] = BBOX.s + 0.3 + Math.random() * 31.4; this.parts[k * 3 + 1] = BBOX.w + 0.3 + Math.random() * 35.4;
    this.age[k] = randomAge ? Math.random() * 80 : 0;
  }

  setShock(s) {
    if (!s) { this.shock.visible = false; return; }
    const ring = r => { const p = []; for (let a = 0; a <= 64; a++) { const t = a / 64 * 2 * Math.PI; p.push(ll2v(s.lat + r * Math.sin(t), s.lon + r * Math.cos(t) / Math.cos(s.lat * D2R), 1.016)); } return p; };
    this.shockRing.geometry.setFromPoints(ring(s.radius)); this.shockR = s; this.ringFn = ring; this.shock.visible = true;
  }

  flyTo(lat, lon, dist = 2.4, dur = 2.6) { this.intro = { t: 0, dur, from: this.camera.position.clone(), to: ll2v(lat, lon, dist) }; }

  frame(dt, opts) {
    if (!this.active) return;
    const now = performance.now();
    if (this.intro) {                                       // smooth great-circle fly-in
      this.intro.t = Math.min(1, this.intro.t + dt / (this.intro.dur || 2.6)); const e = 1 - Math.pow(1 - this.intro.t, 3);
      const a = this.intro.from.clone().normalize(), b = this.intro.to.clone().normalize();
      const len = this.intro.from.length() + (this.intro.to.length() - this.intro.from.length()) * e;
      this.camera.position.copy(a.lerp(b, e).normalize().multiplyScalar(len)); this.camera.lookAt(0, 0, 0);
      if (this.intro.t >= 1) this.intro = null;
    }
    this.controls.autoRotate = this.autoRotate && !this.intro; this.controls.autoRotateSpeed = 0.35;
    this.controls.update();
    if (this.shock.visible && this.shockR) {
      const p = (now - this.t0) / 1600 % 1;
      this.shockPulse.geometry.setFromPoints(this.ringFn(this.shockR.radius * (0.2 + p * 1.1)));
      this.shockPulse.material.opacity = 1 - p;
    }
    this.clouds.mesh.visible = this.shadow.mesh.visible = opts.clouds;
    this.wind.visible = opts.wind && !!this.u; if (this.wind.visible) this.stepWind(dt);
    this.rainMesh.visible = opts.rain && !!this.rainCdf; if (this.rainMesh.visible) this.stepRain(dt);
    this.stepLightning(dt, opts.rain);
    if (this.borders) this.borders.visible = opts.borders;
    this.renderer.render(this.scene, this.camera);
  }

  stepRain(dt) {
    const R = this.rain, pos = this.rainPos, col = this.rainCol, a = new THREE.Vector3(), b = new THREE.Vector3();
    const n = Math.min(this.RN, Math.floor(this.rainTot / 6));
    for (let k = 0; k < this.RN; k++) {
      if (k >= n) { pos.fill(0, k * 6, k * 6 + 6); continue; }
      R[k * 4 + 2] -= dt * R[k * 4 + 3];
      if (R[k * 4 + 2] < 0) this.spawnRain(k);
      const h = R[k * 4 + 2], r0 = 1.002 + h * 0.012;
      ll2v(R[k * 4], R[k * 4 + 1], r0 + 0.0022, a); ll2v(R[k * 4], R[k * 4 + 1], r0, b);
      pos.set([a.x, a.y, a.z, b.x, b.y, b.z], k * 6);
      col.set([0.15, 0.25, 0.4, 0.55, 0.75, 1], k * 6);
    }
    this.rainGeo.attributes.position.needsUpdate = true; this.rainGeo.attributes.color.needsUpdate = true;
  }

  stepLightning(dt, on) {
    for (const b of this.bolts) {
      b.life -= dt; b.s.material.opacity = Math.max(0, b.life / 0.18) * (Math.random() > 0.3 ? 1 : 0.4);
      if (b.life <= 0 && on && this.storm?.length && Math.random() < dt * 2.2) {
        const k = this.storm[Math.floor(Math.random() * this.storm.length)];
        ll2v(LAT0 + Math.floor(k / NLON) + Math.random() - 0.5, LON0 + k % NLON + Math.random() - 0.5, 1.016, b.s.position);
        b.s.scale.setScalar(0.03 + Math.random() * 0.04); b.life = 0.18;
      }
    }
  }

  stepWind(dt) {
    const s = Math.min(dt, 0.05) * 0.4, P = this.parts, pos = this.windPos, col = this.windCol, a = new THREE.Vector3(), b = new THREE.Vector3();
    const dark = this.uni.light.value ? 0.55 : 1;
    for (let k = 0; k < this.N; k++) {
      const lat = P[k * 3], lon = P[k * 3 + 1];
      const gi = lat - LAT0, gj = lon - LON0, u = bilinear(this.u, gi, gj), v = bilinear(this.v, gi, gj), sp = Math.hypot(u, v);
      const dlat = v * s, dlon = u * s / Math.cos(lat * D2R);
      const nlat = lat + dlat, nlon = lon + dlon;
      ll2v(nlat - dlat * 7, nlon - dlon * 7, 1.009, a); ll2v(nlat, nlon, 1.009, b);
      pos.set([a.x, a.y, a.z, b.x, b.y, b.z], k * 6);
      const life = Math.min(1, this.age[k] / 12, (90 - this.age[k]) / 12), br = Math.min(1, 0.25 + sp / 10) * Math.max(0, life) * dark;
      col.set([0, 0, 0, br, br, br], k * 6);
      P[k * 3] = nlat; P[k * 3 + 1] = nlon; this.age[k] += 1;
      if (this.age[k] > 90 || nlat < BBOX.s || nlat > BBOX.n || nlon < BBOX.w || nlon > BBOX.e) this.respawn(k, false);
    }
    this.windGeo.attributes.position.needsUpdate = true; this.windGeo.attributes.color.needsUpdate = true;
  }

  snapshot() { return this.renderer.domElement.toDataURL('image/png'); }
}
export { NLAT, NLON };
