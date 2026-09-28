"""A small mushroom body with the published layout, for checking the memory model's behaviour."""
import json, numpy as np
def synthetic(seed=0, kcs=400, pns=50):
    rng = np.random.default_rng(seed)
    nodes, edges = [], []
    add = lambda b, role, t, cin=(), cout=(), nt='acetylcholine', kc=None: nodes.append(
        {'bodyId': b, 'role': role, 'type': t, 'instance': t, 'compartments_in': list(cin), 'compartments_out': list(cout), 'nt': nt, 'kc_class': kc})
    for i in range(pns): add(1000 + i, 'PN', 'PN')
    for i in range(kcs): add(2000 + i, 'KC', 'KCg-m' if i < kcs // 2 else 'KCab-s', kc='γ' if i < kcs // 2 else 'α/β')
    # compartments: punishment-taught (approach outputs) and reward-taught (avoid outputs)
    layout = [('γ1pedc', 'PPL101', 'glutamate'), ('γ2', 'PPL103', 'gaba'), ('γ4', 'PAM08', 'glutamate'), ('γ5', 'PAM07', 'glutamate'),
              ('α3', 'PPL106', 'acetylcholine'), ('β1', 'PAM10', 'acetylcholine')]
    for j, (comp, dan, nt) in enumerate(layout):
        add(3000 + j, 'MBON', f'MBON{j:02d}', cin=[comp], nt=nt)
        add(4000 + j, 'DAN', dan, cout=[comp], nt='dopamine')
    for i in range(kcs):
        for p in rng.choice(pns, 6, replace=False): edges.append([1000 + int(p), 2000 + i, int(rng.integers(3, 12))])
        gamma = i < kcs // 2
        for j, (comp, _, _) in enumerate(layout):
            if comp.startswith('γ') == gamma: edges.append([2000 + i, 3000 + j, int(rng.integers(2, 8))])
    # loops: approach outputs excite reward teachers, a route for extinction as a new opposing memory
    edges += [[3000, 4002, 20], [3004, 4005, 20]]
    return {'nodes': nodes, 'edges': edges}
