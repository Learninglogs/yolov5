# -- coding: utf-8 --
#pip install moviepy
from moviepy.editor import VideoFileClip
import os

def getMusic(video_name):
    """
    获取指定视频的音频
    :param video_name: 视频名称
    :return: 音频对象
    """
    # 读取有声音的视频文件
    video1 = VideoFileClip(video_name)
    # 返回音频
    audio = video1.audio
    return audio

def addMusic(video_name, audio):
    """实现混流，给video_name添加音频"""
        # 读取没有声音的视频
    video = VideoFileClip(video_name)
    # 设置视频的音频
    print("再调用这里\n")
    video = video.set_audio(audio)
    # 保存新的视频文件
    video.write_videofile( SAVE_DIR)


if __name__ == '__main__':
    # 混流
    SAVE_DIR = "runs/detect/new.mp4"
    input_video_name_dir = "data/images/"
    input_dmvideo_name_dir = "runs/detect/exp/"
    input_video_name = os.listdir(input_video_name_dir)
    input_dmvideo_name = os.listdir(input_dmvideo_name_dir)
    # print(input_video_name_dir+input_video_name[0])
    # print(input_dmvideo_name_dir+input_dmvideo_name[0])
    name = input_video_name_dir+input_video_name[0]
    name2 = input_dmvideo_name_dir+input_dmvideo_name[0]
    #请把要处理的两个视频，放在当前目录下，并且name直接传文件名，能成功。上述代码，传了路径进去，会出现一些问题
    addMusic(name, getMusic(name2))#这样是没问题的

