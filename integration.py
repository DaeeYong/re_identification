'''
Author: Kim Dae_Yong
date: 2024.09.02
last_modified: 2024.09.02

Comment:
겨우 완성했다...후
'''

import cv2
import json
from dragon import dragonV

######################################################################################
video_path = '/media/yong/SAMSUNG1/video/pp009/fast_rear.mp4'
yolo_data_path = './output/pp09_roi2.json'

#Openpose json file folder
json_folder_path = '/media/yong/SAMSUNG1/json/pp009/pp009_omc_walk_fast_rear.mp4/'

re_target_output_path = './output/re_target_2.json'

SAVE = False
VIDEO_SPEED = 50 #ms
######################################################################################

with open(yolo_data_path, 'r', encoding='utf-8') as file:
    yolo_data = json.load(file)
    
pos = dragonV.from_jsonfolder_to_list(json_folder_path)

# 재분류된 Joint Position 저장을 위한 딕셔너리
reclassified_joint_positions = {}

# 비디오 파일 열기
input_video_path = video_path  # 입력 비디오 경로

cap = cv2.VideoCapture(input_video_path)

frame_number = 0

# 마진 설정 (예: 10 픽셀)
margin = 10

# 가장 가까운 프레임을 찾는 함수
def find_closest_frame(target_frame, available_frames):
    closest_frame = min(available_frames, key=lambda x: abs(x - target_frame))
    return closest_frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for person_id, detections in yolo_data.items():
        if frame_number < len(pos):
            frame_pos_data = pos[frame_number] 

            available_frames = [detection[0] for detection in detections]
            closest_frame = find_closest_frame(frame_number, available_frames)

            closest_detection = next(d for d in detections if d[0] == closest_frame)
            _, x1, y1, x2, y2 = closest_detection

            x1 -= margin
            y1 -= margin
            x2 += margin
            y2 += margin

            for person_index, joint_data in enumerate(frame_pos_data):
                person_joint_positions = [frame_number]
                has_joint_inside = False 

                for i in range(0, len(joint_data), 2):
                    x = int(joint_data[i])
                    y = int(joint_data[i + 1])

                    # 관절이 조정된 바운딩 박스 내에 있는지 확인
                    if x1 < x < x2 and y1 < y < y2:
                        person_joint_positions.extend([x, y])
                        has_joint_inside = True

                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                        cv2.putText(frame, f"ID: {person_id}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        person_joint_positions.extend([0, 0])

                if has_joint_inside:
                    if person_id not in reclassified_joint_positions:
                        reclassified_joint_positions[person_id] = []
                    reclassified_joint_positions[person_id].append(person_joint_positions)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 프레임을 화면에 표시
    cv2.imshow('Video', frame)

    if cv2.waitKey(VIDEO_SPEED) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

if SAVE == True:
    # 재분류된 Joint Position을 JSON 파일로 저장
    with open(re_target_output_path, 'w') as outfile:
        json.dump(reclassified_joint_positions, outfile, indent=4)

    print(f"Reclassified joint positions saved to {re_target_output_path}") 