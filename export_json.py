"""
   file: export_json.py
   subject : count fish
   Author : AVS7
   Creation : 7/11/2022
   Last Update : 7/11/2022
   Update Note:
        7/11/2022 : Creation


"""
import glob
import json
import os

import numpy as np

#from video import video as vd

def initialize_json(file_name):
    if os.path.exists(file_name):  # if file exists, delete it
        os.remove(file_name)

    with open(file_name, 'w+') as file:
        data = "{\"results\": []}"
        file.write(data)
        file.close()

def export_json(export_file_name, file_name, exit_frame, enter_frame, sum, fish_count_frames):
    """
    export json
    Args:

        list_video: list of videos
        path: path to folder containing videos
        file_name: name of the file

    Returns:

    """
    with open(export_file_name, 'r+') as file:
        file_data = json.load(file)
        file_data["results"].append({
            "video": file_name,
            "fish_count_frames": fish_count_frames,
            "enter_frames": enter_frame,
            "exit_frames": exit_frame,
            "fish_count": sum,
        })
        file.seek(0)
        json.dump(file_data,file, indent=4)
        #file.write(',\n')

