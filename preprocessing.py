'''
Author: Kim Dae_Yong
date: 2024.09.02
last_modified: 2024.09.02

Comment:
겨우 완성했다...
'''

from dragon import dragonY
from collections import defaultdict
import json
import os

####################################################################

input_video_path = '/media/yong/SAMSUNG1/video/pp009/fast_rear.mp4'
output_roi_path = './output/pp09_roi2.json'

#그대로 사용하는 것을 추천
tracking_num = 2

##################################################################

results = dragonY.get_results_tracking_data_from_video(input_video_path)
results_dict = dragonY.get_all_frame_data_list_from_yolo_results(results)

track_list = defaultdict(lambda: [])
for frame_idx in range(len(results_dict)):
    for element in results_dict[frame_idx]:
        if element['id'] == -1: continue

        track_id = element['id']
        box = element['xywh']
        x, y, w, h = box
        track = track_list[track_id]
        track.append([frame_idx, int(x), int(y), int(w), int(h)])

sorted_items = sorted(track_list.items(), key=lambda item: len(item[1]), reverse=True)
roi_data = dict(sorted_items[:tracking_num])

# Coordinate transform: Yolo xywh -> x1 y1 x2 y2
for yolo_idx in roi_data:
    for idx, ixywh_data in enumerate(roi_data[yolo_idx]):
        i = ixywh_data[0]
        x = ixywh_data[1]
        y = ixywh_data[2]
        w = ixywh_data[3]
        h = ixywh_data[4]
        
        pt1_x, pt1_y, pt2_x, pt2_y = dragonY.coord_transform(x,y,w,h)
        roi_data[yolo_idx][idx] = [i, pt1_x, pt1_y, pt2_x, pt2_y]
        
#result save
with open(output_roi_path, 'w') as json_file:
    json.dump(roi_data, json_file, indent=4)
