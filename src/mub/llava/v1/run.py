"""V1: collect, three-fold fit/select layer, then calibrate sign."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image
from mub.llava.core import CONFIG, RESULTS, config_file, ensure_manifest, resolve, write_json
from mub.llava.adapter import check_backend, load_engine
from mub.llava.data import load_split
from mub.llava import protocol as current_protocol
from mub.llava.protocol import training_options


def manifest(config, rows, protocol=current_protocol):
    return {'training_protocol': protocol.TRAINING_PROTOCOL, 'config': config, 'items': rows}


def collect(config, rows, output, engine, protocol=current_protocol):
    output = Path(output)
    for row in rows:
        pid = row['pair_id']
        record_path = output / 'records' / f'{pid}.json'
        state_path = output / 'activations' / f'{pid}.npz'
        if record_path.exists() and state_path.exists():
            continue
        if record_path.exists() != state_path.exists():
            raise ValueError(f'Incomplete collection cache: {pid}; use another --output')
        record, states = {'pair_id': pid}, []
        options = training_options(row)
        with Image.open(resolve(row['image_path'])) as handle:
            image = handle.convert('RGB')
        try:
            inputs = engine.prepare(protocol.training_messages(row, image), image)
            record['initial'] = engine.score(inputs, options=options)[0]
            for arm in ['dplus', 'dminus']:
                inputs = engine.prepare(protocol.training_messages(
                    row, image, arm, initial=record['initial']['answer_text']), image)
                record[arm], captured = engine.score(inputs, options=options, capture=config['decoder_layers'])
                states.append(np.stack([captured[str(layer)] for layer in config['decoder_layers']]))
        finally:
            image.close()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, states=np.stack(states))
        write_json(record_path, record)
        print(f'[collect] {pid}', flush=True)


def run(config, output, stage='all', protocol=current_protocol, model='llava', engine=None):
    from mub.llava.v1.extract import fit
    from mub.llava.v1.calibrate import calibrate
    check_backend(config, model)
    rows = load_split('train')
    if config['outer_folds'] != 3:
        raise ValueError('Use the same three-fold design as steering')
    ensure_manifest(output, manifest(config, rows, protocol))
    if stage in ('all', 'collect'):
        if any(not (Path(output) / 'records' / f"{r['pair_id']}.json").exists()
               or not (Path(output) / 'activations' / f"{r['pair_id']}.npz").exists() for r in rows):
            if engine is None:
                engine = load_engine(config, model)
            write_json(Path(output) / 'model.json', {**engine.metadata(), 'protocol': protocol.TRAINING_PROTOCOL})
        collect(config, rows, output, engine, protocol)
    result = None
    if stage in ('all', 'fit'):
        result = fit(config, rows, output)
    if stage in ('all', 'calibrate'):
        if engine is None and any(not (Path(output) / 'calibration/records' / f"{r['pair_id']}.json").exists() for r in rows):
            engine = load_engine(config, model)
        result = calibrate(config, rows, output, engine, protocol)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--stage', choices=['all', 'collect', 'fit', 'calibrate'], default='all')
    args = parser.parse_args(argv)
    result = run(config_file(args.config or CONFIG, args.model_path),
                 resolve(args.output or RESULTS / 'v1'), args.stage, model='llava')
    if result is not None:
        print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
