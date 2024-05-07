import subprocess
import cv2
import matplotlib.pyplot as plt
import numpy as np


def run(video, compressed_video, white_level, black_level):
    # Сжатие видео
    video_compression(video, compressed_video)

    # Создаём объекты для работы с видео
    cap = cv2.VideoCapture(video)
    cap_compressed = cv2.VideoCapture(compressed_video)

    # Разность кадров
    frame_difference(cap, cap_compressed, white_level, black_level)

    # Освобождаем ресурсы
    cap.release()
    cap_compressed.release()


def video_compression(video_path, compressed_video_path):
    bitrate = '2M'
    command = ['ffmpeg', '-hide_banner', '-i', video_path,
               '-b:v', bitrate, '-minrate', bitrate, '-maxrate', bitrate, '-bufsize', bitrate,
               compressed_video_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def frame_difference(cap, cap_compressed, white_level, black_level):
    # Получаем данные о видео
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Хранение значений разницы между кадрами
    frame_differences = np.zeros(frame_count)

    # Максимальная разница и таймкод
    max_difference = 0
    max_difference_frame = 0

    # Обработка всех кадров
    for i in range(frame_count):
        ret, frame = cap.read()
        ret_compressed, frame_compressed = cap_compressed.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_compressed = cv2.cvtColor(frame_compressed, cv2.COLOR_RGB2GRAY)

        # Повышение контрастности
        frame_compressed = (((frame_compressed - np.min(frame_compressed)) *
                             ((white_level - black_level) /
                              (np.max(frame_compressed) - np.min(frame_compressed)))) +
                            black_level)

        # Вычисление средней разницы для текущего кадра
        frame_differences[i] = np.mean(np.abs(frame - frame_compressed))

        # Максимальная разница и её кадр
        if frame_differences[i] > max_difference:
            max_difference = frame_differences[i]
            max_difference_frame = i

    # Строим график
    graph(frame_count, frame_differences, max_difference_frame, max_difference)


def graph(frame_count, frame_differences, max_difference_frame, max_difference):
    # Вывод графика средней разницы и таймкода максимальной ошибки
    plt.figure(figsize=(12, 6))
    plt.plot(range(frame_count), frame_differences)
    plt.scatter(max_difference_frame, max_difference, color='r', label='Максимальная ошибка')
    plt.xlabel('Кадры')
    plt.ylabel('Средняя разница')
    plt.title('Оценка качества сжатия видео')
    plt.legend()
    plt.show()
