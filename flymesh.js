'use strict';
/* Procedural Drosophila mesh: thorax, segmented abdomen, head with faceted compound eyes, antennae,
   veined wings, halteres, proboscis, bristles, and six legs of coxa, femur, tibia and tarsus at the
   segment lengths in bodyplan.py. An illustration, not a scanned specimen: proportions are drawn by
   eye, and only the leg joints have measured motor neurons behind them. */
(function (root) {
  const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
  const SEG = { coxa: 0.30, femur: 0.75, tibia: 0.75, tarsus: 0.55 };   // mm, as in bodyplan.py
  const HIP = { F: 0.55, M: 0.0, H: -0.55 };
  const PART_NAMES = {
    thorax: 'Thorax', abdomen: 'Abdomen', head: 'Head', left_eye: 'Left compound eye', right_eye: 'Right compound eye',
    left_antenna: 'Left antenna', right_antenna: 'Right antenna', left_wing: 'Left wing', right_wing: 'Right wing',
    left_haltere: 'Left haltere', right_haltere: 'Right haltere', proboscis: 'Proboscis',
    LF: 'Left front leg', LM: 'Left middle leg', LH: 'Left hind leg',
    RF: 'Right front leg', RM: 'Right middle leg', RH: 'Right hind leg',
  };
  const BRAIN_DRIVEN = new Set(LEGS);   // the rest have no motor neurons in this subset

  const mat = (colour, extra = {}) => new THREE.MeshLambertMaterial(Object.assign({ color: colour }, extra));
  function ellipsoid(parent, scale, position, colour, extra) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), mat(colour, extra));
    m.scale.set(...scale); m.position.set(...position); parent.add(m); return m;
  }
  function rod(parent, from, to, radius, colour) {
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to), d = b.clone().sub(a);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.7, radius, d.length(), 6), mat(colour));
    m.position.copy(a.clone().add(b).multiplyScalar(0.5));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
    parent.add(m); return m;
  }
  function limb(length, radius, colour) {      // a segment lying along +x from its joint
    const g = new THREE.Group();
    const m = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.72, radius, length, 6), mat(colour));
    m.rotation.z = -Math.PI / 2; m.position.x = length / 2; g.add(m);
    return g;
  }

  function build(colour) {
    const root = new THREE.Group(), parts = {}, legs = {};
    const part = (name, position, parent) => {
      const g = new THREE.Group(); g.position.set(...position); g.userData.part = name; g.userData.base = g.position.clone();
      parts[name] = g; (parent || root).add(g); return g;
    };
    const thorax = part('thorax', [0, 0, 0]);
    ellipsoid(thorax, [0.52, 0.40, 0.38], [0, 0, 0], colour);
    ellipsoid(thorax, [0.12, 0.05, 0.3], [0.08, 0.36, 0], 0xf0f4e6);            // scutellum highlight
    for (let i = 0; i < 22; i++) {                                              // thoracic bristles
      const a = i * 2.39996, x = 0.34 * Math.cos(a), z = 0.3 * Math.sin(a);
      rod(thorax, [x, 0.2, z], [x * 1.25, 0.5 + (i % 3) * 0.05, z * 1.3], 0.006, 0x3f3728);
    }
    const abdomen = part('abdomen', [-0.62, -0.02, 0]);
    for (let i = 0; i < 6; i++) {
      const s = 0.36 * (1 - i * 0.11);
      ellipsoid(abdomen, [0.17, s * 0.82, s], [0.1 - i * 0.18, -i * 0.03, 0], i % 2 ? 0x4a3a27 : colour);
    }
    const head = part('head', [0.62, 0.05, 0]);
    ellipsoid(head, [0.3, 0.3, 0.3], [0, 0, 0], 0xbd9358);
    for (const side of [-1, 1]) {
      const key = side < 0 ? 'left' : 'right';
      const eye = part(key + '_eye', [0.06, 0.04, side * 0.22], head);
      ellipsoid(eye, [0.21, 0.25, 0.15], [0, 0, 0], 0x8d2e1b);
      const facets = new THREE.InstancedMesh(new THREE.SphereGeometry(0.026, 5, 4), mat(0xc0502a), 90), dummy = new THREE.Object3D();
      for (let i = 0; i < 90; i++) {                                            // faceted surface
        const y = 1 - 2 * (i + 0.5) / 90, r = Math.sqrt(Math.max(0, 1 - y * y)), a = i * 2.39996;
        dummy.position.set(0.21 * r * Math.cos(a), 0.25 * y, 0.15 * r * Math.sin(a));
        dummy.updateMatrix(); facets.setMatrixAt(i, dummy.matrix);
      }
      eye.add(facets);
      const antenna = part(key + '_antenna', [0.22, 0.02, side * 0.1], head);
      rod(antenna, [0, 0, 0], [0.16, 0.08, side * 0.06], 0.03, 0x5a3f27);
      ellipsoid(antenna, [0.06, 0.05, 0.05], [0.16, 0.08, side * 0.06], 0x744c2d);
      rod(antenna, [0.16, 0.08, side * 0.06], [0.3, 0.2, side * 0.1], 0.01, 0xa08a68);   // arista
      const wing = part(key + '_wing', [-0.12, 0.34, side * 0.2]);
      const shape = new THREE.Shape();
      shape.moveTo(0, 0);
      shape.bezierCurveTo(0.2, 0.15, 0.3, 0.9, -0.12, 1.7);
      shape.bezierCurveTo(-0.6, 2.1, -0.9, 1.35, -0.8, 0.82);
      shape.bezierCurveTo(-0.56, 0.37, -0.22, 0.11, 0, 0);
      const blade = new THREE.ShapeGeometry(shape, 16);
      blade.rotateX(Math.PI / 2); if (side < 0) blade.scale(1, 1, -1);
      wing.add(new THREE.Mesh(blade, mat(0xb5d1c8, { transparent: true, opacity: 0.34, side: THREE.DoubleSide, depthWrite: false })));
      for (const [x, z] of [[-0.08, 1.64], [-0.38, 1.66], [-0.66, 1.22], [-0.72, 0.86]]) rod(wing, [0, 0.007, 0], [x, 0.007, z * side], 0.008, 0x779189);
      const haltere = part(key + '_haltere', [-0.34, 0.02, side * 0.26]);
      rod(haltere, [0, 0, 0], [-0.1, 0.05, side * 0.26], 0.02, 0xaf925d);
      ellipsoid(haltere, [0.06, 0.05, 0.06], [-0.1, 0.05, side * 0.26], 0xd9c096);
    }
    const proboscis = part('proboscis', [0.78, -0.12, 0]);
    rod(proboscis, [0, 0, 0], [0.1, -0.16, 0], 0.04, 0x9e7348);
    ellipsoid(proboscis, [0.09, 0.03, 0.07], [0.1, -0.16, 0], 0xc69260);
    for (const name of LEGS) {
      const side = name[0] === 'L' ? -1 : 1;
      const coxa = part(name, [HIP[name[1]], -0.2, side * 0.26]);
      coxa.add(limb(SEG.coxa, 0.05, 0x6b5130));
      const femur = limb(SEG.femur, 0.042, 0x7b5c35); femur.position.x = SEG.coxa; coxa.add(femur);
      const tibia = limb(SEG.tibia, 0.034, 0x6b5130); tibia.position.x = SEG.femur; femur.add(tibia);
      const tarsus = limb(SEG.tarsus, 0.026, 0x8a6a3c); tarsus.position.x = SEG.tibia; tibia.add(tarsus);
      const foot = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), mat(0xc2e899));
      foot.position.x = SEG.tarsus; tarsus.add(foot);
      legs[name] = { coxa, femur, tibia, tarsus, foot, side };
    }
    return { root, parts, legs };
  }

  root.FlyMesh = { build, LEGS, SEG, HIP, PART_NAMES, BRAIN_DRIVEN, mat };
})(typeof window !== 'undefined' ? window : globalThis);
