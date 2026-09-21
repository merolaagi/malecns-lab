"""Optional: download official raw tables to reproduce the bundled subset (~1.1 GB)."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from build_dataset import FILES, SOURCE, build

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for key, name in FILES.items():
        target = args.directory / (key + '.feather')
        if target.exists():
            print('Using existing', target, flush=True)
            continue
        print('Downloading', name, flush=True)
        partial = target.with_suffix('.partial')
        with urlopen(SOURCE + name, timeout=60) as response, partial.open('wb') as output:
            while block := response.read(8 * 1024 * 1024): output.write(block)
        partial.replace(target)
    build(args.directory)
