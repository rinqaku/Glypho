#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    registry = json.loads((ROOT / '.github' / 'model-registry.json').read_text(encoding='utf-8'))
    rust = (ROOT / 'crates' / 'glypho' / 'src' / 'onnx.rs').read_text(encoding='utf-8')
    web = (ROOT / 'web' / 'src' / 'engine' / 'models.ts').read_text(encoding='utf-8')
    models = {model['id']: model for model in registry['models']}
    selected = {
        model_id
        for profile in registry['profiles'].values()
        for model_id in (profile['detector'], profile['recognizer'], *profile['specialists'])
    }

    errors = []
    missing = sorted(selected - models.keys())
    if missing:
        errors.append(f'profiles reference missing models: {", ".join(missing)}')

    for model_id in sorted(selected & models.keys()):
        model = models[model_id]
        tokens = [model['repository'], model['revision']]
        for artifact in model['artifacts']:
            tokens.append(artifact['sha256'])
        for target, source in (('Rust', rust), ('Web', web)):
            absent = [token for token in tokens if token not in source]
            compact = source.replace('_', '')
            absent.extend(
                str(artifact['bytes'])
                for artifact in model['artifacts']
                if str(artifact['bytes']) not in compact
            )
            if absent:
                errors.append(f'{model_id} is incomplete in {target}: {", ".join(absent)}')

    if errors:
        print('\n'.join(errors))
        return 1
    print(f'{len(selected)} model definitions match Rust and Web')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
