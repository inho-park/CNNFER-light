import os
import sys

def convert_webm_mp4_subprocess(input_file, output_file):
    command_path = 'set path=%path%;' + os.getcwd() + r'\python_src\webmToMp4\ffmpeg\bin; &'
    command_ffm = 'ffmpeg -i ' + input_file + ' ' + output_file
    os.system(command_path + command_ffm)

# convert_webm_mp4_subprocess('../video/test4.webm', '../video/test4.mp4')
def main(argv):
    convert_webm_mp4_subprocess(argv[1], argv[2])


if __name__ == "__main__":
    main(sys.argv)
