"""Run InternVL3-8B MUB post-answer V1 extraction/calibration and V2 evaluation."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mub.internvl.core import CONFIG, RESULTS, MODEL, config_file, resolve
from mub.internvl.adapter import check_backend
from mub.internvl import protocol as current_protocol


def main(argv=None, *, protocol=current_protocol):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'validate', 'smoke', 'collect', 'fit', 'calibrate', 'evaluate'], default='all')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args(argv)
    config, output = config_file(args.config, args.model_path), resolve(args.output)
    check_backend(config)
    if args.stage in ('validate', 'smoke'):
        from mub.internvl.validate import validate, smoke
        print(validate(config, MODEL))
        if args.stage == 'smoke':
            print(smoke(config, MODEL, protocol))
        return 0
    if args.stage != 'evaluate':
        from mub.internvl.v1.run import run
        run(config, output / 'v1', args.stage, protocol=protocol)
    if args.stage in ('all', 'evaluate'):
        from mub.internvl.v2.run import run
        print(run(config, output / 'v1', output / 'v2', protocol=protocol))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
