import os
import re
import argparse
import subprocess
from faster_whisper import WhisperModel

def main(media_folder, whisper_model, language, device):

    models_name = os.listdir("models")
    name=whisper_model.split("/")[1] 

    if name not in models_name:
        st = ', '.join(models_name)
        raise Exception(f'Модель {name} не найдена. Доступные модели: {st}')
    # Аудио и видео файлы с указанными расширениями будут обработаны
    audio_exts = ['mp3', 'aac', 'ogg', 'wav']
    video_exts = ['mp4', 'avi', 'mov', 'mkv']

    print(f'Поиск в папке "{media_folder}"')
    files = [file for file in os.listdir(media_folder) if match_ext(file, audio_exts + video_exts)]
    print(f'Найдено {len(files)} файлов:')
    for filename in files: 
        print(filename)
    
    for filename in files:
        print(f'\n\nОбрабатываю {filename}')
        media_file = os.path.join(media_folder, filename)
        if filename.split('.')[-1] in video_exts:
            audio_file = extract_audio(media_file)
            process_audiofile(fname=media_file, whisper_model=whisper_model, original_file=media_file, language=language, device = device)
            os.remove(audio_file)  # Удаляем временный аудиофайл после обработки
        else:
            process_audiofile(fname=media_file, whisper_model=whisper_model,  language=language, device=device)

def match_ext(filename, extensions):
    return filename.split('.')[-1] in extensions

def extract_audio(video_file):
    audio_file = video_file.rsplit('.', 1)[0] + '.wav'
    command = [
        'ffmpeg',
        '-i', video_file,
        '-q:a', '0',
        '-map', 'a',
        audio_file
    ]
    subprocess.run(command, check=True)
    return audio_file

def process_audiofile(fname, whisper_model, original_file=None, language='auto', device='cpu'):
    fext = fname.split('.')[-1]
    fname_noext = fname[:-(len(fext)+1)]

    # Загрузка модели faster-whisper
    if device == 'cpu':
        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    if device == 'gpu':
        model = WhisperModel(whisper_model, device="cuda", compute_type="float16")



    if language == 'auto':
        # Детектирование языка в файле
        segments, info = model.transcribe(fname, beam_size=5, language=None)
        language = info.language
        print(f"Определен язык: {language} Точность: {info.language_probability}")
    else:
        # Транскрибирование с указанным языком
        segments, info = model.transcribe(fname, beam_size=5, language=language, vad_filter=True)

    whisper_model_name=whisper_model.split("/")[1]
    # Создаем файл с таймкодами
    with open(fname_noext + '_timecode_' + whisper_model_name + '.txt', 'w', encoding='UTF-8') as f:
        for segment in segments:
            timecode_sec = int(segment.start)
            hh = timecode_sec // 3600
            mm = (timecode_sec % 3600) // 60
            ss = timecode_sec % 60
            timecode = f'[{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}]'
            text = segment.text.strip()
            f.write(f'{timecode} {text}\n')

#    # Создаем файл с "сырым" текстом
#    rawtext = ' '.join([segment.text.strip() for segment in segments])
#    rawtext = re.sub(" +", " ", rawtext)
#
#    with open(fname_noext + '.txt', 'w', encoding='UTF-8') as f:
#        f.write(rawtext)

#    # Создаем SRT файл, если есть исходный файл
#    if original_file:
#        srt_file = fname_noext + '.srt'
#        with open(srt_file, 'w', encoding='UTF-8') as f:
#            for idx, segment in enumerate(segments):
#                start_time = format_time(segment.start)
#                end_time = format_time(segment.end)
#                text = segment.text.strip()
#                f.write(f"{idx + 1}\n{start_time} --> {end_time}\n{text}\n\n")

def format_time(seconds):
    hh = int(seconds) // 3600
    mm = (int(seconds) % 3600) // 60
    ss = int(seconds) % 60
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio and video files for transcription using faster-whisper.')
    parser.add_argument('media_folder', type=str, help='Folder containing media files to process')
    parser.add_argument('whisper_model', type=str, help='Whisper model to use for transcription (e.g. base, medium, large)')
    parser.add_argument('--language', type=str, default='auto', help='Source language code for transcription (default: auto)')
    parser.add_argument('--device', type=str, default='cpu', help='Source cpu or gpu (default: auto)')

    args = parser.parse_args()

    # Calling main() function with media folder, whisper model, and additional arguments
    main(args.media_folder, args.whisper_model, args.language, args.device)
