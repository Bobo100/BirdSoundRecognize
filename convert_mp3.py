# Convert a folder of mp3 recordings (one subfolder per species) to wav.
# Usage: python convert_mp3.py <mp3_folder> [--output data/audio]
# Requires ffmpeg on PATH (used by pydub).

import argparse
import os

from pydub import AudioSegment

import config as cfg


def mp3_to_wav(input_folder, output_folder):
    error_log = os.path.join(output_folder, 'conversion_errors.txt')

    for root, _, files in os.walk(input_folder):
        # Files directly in the input folder have no species folder
        if os.path.samefile(root, input_folder):
            if any(f.lower().endswith('.mp3') for f in files):
                print(f'skipped mp3 files directly in {input_folder}: put them in a species subfolder')
            continue

        # Mirror the species subfolder in the output folder
        target_dir = os.path.join(output_folder, os.path.relpath(root, input_folder))

        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() != '.mp3':
                continue

            os.makedirs(target_dir, exist_ok=True)
            new_file_path = os.path.join(target_dir, stem + '.wav')
            if os.path.exists(new_file_path):
                continue

            try:
                sound = AudioSegment.from_mp3(os.path.join(root, name))
                print(f'sound duration_seconds = {sound.duration_seconds}')
                sound.export(new_file_path, format='wav')
                print(new_file_path + ' done')
            except Exception as e:
                print(f'failed: {name} ({e})')
                # Remove a partial wav so the next run retries it
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                with open(error_log, 'a', encoding='utf-8') as f:
                    f.write(os.path.join(root, name) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mp3 recordings to wav')
    parser.add_argument('input', help='folder with one mp3 subfolder per species')
    parser.add_argument('--output', default=cfg.AUDIO_PATH, help='wav output folder (default: data/audio)')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        parser.error(f'input folder not found: {args.input}')

    print('\n==Export Start==')
    mp3_to_wav(args.input, args.output)
    print('done')
