import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description='Convert webm files of a specific directory to mp3 using ffmpeg')

parser.add_argument('--webm_path',
                    action='store',
                    type=str,
                    required=True,
                    help='The path of webm files')

parser.add_argument('--mp3_path',
                    action='store',
                    type=str,
                    required=False,
                    help='The path of mp3 files')

args = parser.parse_args()
# input_path = args.Path

if not os.path.isdir(args.webm_path):
    print(f'The webm path "{args.webm_path}" does not exist')
    sys.exit()

if args.mp3_path is None:
    args.mp3_path = args.webm_path

elif not os.path.isdir(args.mp3_path):
    os.makedirs(args.mp3_path)

for file in os.listdir(args.webm_path):
    webmFile = os.path.join(args.webm_path, file)
    mp3File = os.path.join(args.mp3_path, file).replace('webm', 'mp3')
    command = f'ffmpeg -i \"{webmFile}\" -vn -ab 128k -ar 48000 -y \"{mp3File}\"'
    subprocess.call(command, shell=True)
