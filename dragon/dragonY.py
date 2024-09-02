import json
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
'''
case1 : 특정 프레임에 아무것도 없는 경우
case2 : 

format)
[
    [
    {"id" : int, "xywh" : [int int int int]},
    {"id" : int, "xywh" : [int int int int]}
    ],
    [
    ...
    ]
]
'''

def get_each_id_data_from_yolo_result(results_list):
    track_list = defaultdict(lambda: [])

    for frame_idx in range(len(results_list)):
        for element in results_list[frame_idx]:
            if element['id'] == -1: continue

            track_id = element['id']
            box = element['xywh']
            x, y, w, h = box
            track = track_list[track_id]
            track.append([frame_idx, int(x), int(y), int(w), int(h)])
    
    return track_list

def get_all_frame_data_list_from_yolo_results(results):
    all_frame_data_list = []
    len_frame = len(results)
    for now_frame_idx in range(0, len_frame):
        #현재 프레임에 대한 데이터를 담기위한 리스트
        now_frame_data = []
        #현재 프레임 정보
        boxes = results[now_frame_idx].boxes
        #xywh 좌표정보
        xywh_list = boxes.xywh.tolist()
        #cls 리스트
        cls_list = boxes.cls.int().tolist()
        #현재 물체에 대한 id
        track_id_list = boxes.id
        #id가 없는 경우 [{id : -1, xywh : []}]
        if (track_id_list == None):
            tmp_dir = {}
            tmp_dir['id'] = -1
            tmp_dir['xywh'] = []
            now_frame_data.append(tmp_dir)
            all_frame_data_list.append(now_frame_data)
            continue

        #id 목록들은 list 타입으로 변환
        track_id_list = track_id_list.int().tolist()

        #하나의 frame box에서 데이터 추출
        for cls, track_id, xywh in zip(cls_list, track_id_list, xywh_list):
            #사람이 아니면 패스
            if(cls != 0):
                continue

            #사람인 경우
            data_frame = {}
            data_frame['id'] = track_id
            data_frame['xywh'] = xywh
            #현재 프레임 데이터에 추가
            now_frame_data.append(data_frame)
        
        all_frame_data_list.append(now_frame_data)
    return all_frame_data_list

# all_frame_data -> save json
def save_all_frame_data_list_as_json(all_frame_data_list, json_path:str):
    # JSON 파일에 리스트 내용 저장
    with open(json_path, 'w') as f:
        json.dump(all_frame_data_list, f)

#json -> all_frame_data
def get_all_frame_data_from_json(json_path:str):
    with open(json_path, 'r') as f:
        all_frame_data_list = json.load(f)
    
    return all_frame_data_list

def get_results_tracking_data_from_video(video_path):
    results = model.track(source=video_path, tracker='bytetrack.yaml', save = False,
                        persist=True)
    return results

'''
currnet_frame_num : zero-based idx
'''
def get_xywh_from_all_frame_data(all_frame_data_list, current_frame_idx, id_idx):
        
        #data가 비어있는 프레임인 경우
        if(all_frame_data_list[current_frame_idx][0]['id'] == -1):
            return -1, -1, -1, -1
        
        center_x = (all_frame_data_list[current_frame_idx][id_idx]['xywh'][0])
        center_y = (all_frame_data_list[current_frame_idx][id_idx]['xywh'][1])
        w = (all_frame_data_list[current_frame_idx][id_idx]['xywh'][2])
        h = (all_frame_data_list[current_frame_idx][id_idx]['xywh'][3])

        x = round(center_x - w/2)
        y = round(center_y - h/2)

        return x, y, round(w), round(h)

'''
xywh는 top-left 기준
'''
def get_linear_margin_to_xywh(x, y, w, h, margin_value):
    margin_x = x - margin_value
    margin_y = y - margin_value
    margin_w = w + 2 * margin_value
    margin_h = h + 2 * margin_value

    return margin_x, margin_y, margin_w, margin_h
'''
all_frame_data_list:list 에서의 xywh 데이터는 소수점이기 때문에 처리 필요.
'''

def coord_transform(x,y,w,h):
    pt1_x = round(x - w/2)
    pt1_y = round(y - h/2)

    pt2_x = round(x + w/2)
    pt2_y = round(y + h/2)

    return pt1_x, pt1_y, pt2_x, pt2_y