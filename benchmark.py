"""Reproducible software experiments, NOT biological performance benchmarks."""
import json
from datetime import datetime, timezone
from model import Circuit, BASE, CONDITIONS, DEFAULT

def main():
    c = Circuit()
    rows = []
    for condition in CONDITIONS:
        for seed in range(1, 6):
            r = c.simulate(condition=condition, seed=seed)
            rows.append({'parameters': r['parameters'], 'metrics': r['metrics']})
    sensitivity = []
    for gain in [7, 10, 14]:
        r = c.simulate(gain=gain)
        sensitivity.append({'parameters': r['parameters'], 'metrics': r['metrics']})
    report = {'created': datetime.now(timezone.utc).isoformat(), 'dataset': c.data['dataset'],
              'source_files': c.data['files'], 'defaults': DEFAULT,
              'caveat': 'Exploratory synthetic model outputs. Parameters have not been fitted to biological recordings. Gain 10 was chosen after inspecting gains 7,10,14,18,20 for non-saturated motor activity, not for agreement with biology. Feedback 2 was selected to activate the engineered sensory loop after lower gains had no effect. No scientific novelty or biological validation is established.',
              'runs': rows, 'gain_sensitivity': sensitivity}
    (BASE / 'data/benchmark.json').write_text(json.dumps(report, indent=2))
    for condition in CONDITIONS:
        selected = [r['metrics'] for r in rows if r['parameters']['condition'] == condition]
        print(condition, 'motor Hz', round(sum(r['motor_mean_hz'] for r in selected) / len(selected), 3),
              'path mm', round(sum(r['path_mm'] for r in selected) / len(selected), 3))

if __name__ == '__main__': main()
