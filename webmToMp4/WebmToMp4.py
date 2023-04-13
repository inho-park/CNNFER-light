import os


def convert_webm_mp4_subprocess(input_file, output_file):
    command_path = 'set path=%path%;' + os.getcwd() + r'\ffmpeg\bin; & '
    command_ffm = 'ffmpeg -i ' + input_file + ' ' + output_file
    print(command_path + command_ffm)
    os.system(command_path + command_ffm)


convert_webm_mp4_subprocess('../webm/emotion_sample2.webm', '../video/test.mp4')

# ffm.convert_webm_mp4_module('../webm/emotion_sample2.webm', '../video/test.mp4')