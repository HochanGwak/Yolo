from fileinput import filename
import re
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, session
import numpy as np
import mimetypes
import cv2
import json
import os
import argparse
import datetime
import flask
import time
import facenet
import detect_face
import tensorflow as tf
import pickle
from PIL import Image
from werkzeug.utils import secure_filename


DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'


modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow object detection API')

    parser.add_argument('--debug', dest='debug',
                        help='Run in debug mode.',
                        required=False, action='store_true', default=True)

    parser.add_argument('--port', dest='port',
                        help='Port to run on.', type=int,
                        required=False, default=DEFAULT_PORT)

    parser.add_argument('--host', dest='host',
                        help='Host to run on, set to 0.0.0.0 for remote access', type=str,
                        required=False, default=DEFAULT_HOST)

    args = parser.parse_args()
    return args

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg','mp4'])


#웹 서비스
@app.route('/')
def upload():
    return render_template('upload.html')

# DRAW_REC : 바운딩 박스를 그리고싶으면 True
def run_model(video, DRAW_BOX):
    frame_list = []
    
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')
            
            video_capture = cv2.VideoCapture(video)
            print('Start Recognition')
            # 학생 이름(set)
            member_names = set(HumanNames)
            # 출석부(list)
            attendance_list = []
            
            tm = time.time()
            
            # 비디오 총 프레임 수 구하기
            vid_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 프레임 수 만큼 반복
            for frame_cnt in range(vid_length):
                # 프레임마다 추론하기 때문에 프레임수에 제한을 둔다.
                # yu영상을 예로들면 영상은 약 33,030프레임이므로 서비스하기 쉽지 않다.
                if frame_cnt > 900:
                    break
                # 서비스 속도의 개선을 위해 영상을 출력하지 않을 때(DRAW_BOX=False) 
                # 프레임을 몇개씩 스킵해서 detection하도록 한다.
                if (not DRAW_BOX) & (frame_cnt % 3 != 0):
                    continue
                
                ret, frame = video_capture.read()
                
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                
                # 결국 이 한줄이 서비스가 오래걸리는 이유인데..
                # 함수의 detect_face함수의 조정으로 개선이 필요해보임(물론 실제로 서비스할때 개선이 필요한 것)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities>0.87:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        # 점찍는 print 때문에 줄바꿈용도로 작성
                                        print()
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        if DRAW_BOX:
                                            cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                            cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 0), thickness=1, lineType=1)
                                        # 출석자는 출석부에 기록됨
                                        attendance_list.append(HumanNames[best_class_indices[0]])
                                        
                            else :
                                if DRAW_BOX:
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                    cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                    cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 0), thickness=1, lineType=1)
                        except:
                            print("error")
                        
                endtimer = time.time()
                # 너무 많아서 주석처리
                # print('[time log]now-time', endtimer-tm)
                print('.', end = '')

                ret, buffer = cv2.imencode('.jpg', frame)
                frame1 = buffer.tobytes()
                # 출력할 프레임 저장
                frame_list.append(frame1)
                
                
    # 영상 종료시 페이지에 출력할 데이터 확인
    # 출석하지 않은 사람은 기억해둔다.
    absent_names = list(set(HumanNames).difference(set(attendance_list)))
    
    if len(absent_names) == 0:
        member_check = '전원출석'
        print(f'[yolo log] 결석자 이름 : {absent_names}')
        print(f'[yolo log] 출력 결과 : {member_check}')
        
    else:
        member_check = '결석 인원 : '
        print(f'[yolo log] 결석 인원 : {absent_names}')
        for name in absent_names:
            member_check = member_check + name + ' '
            
    return frame_list, member_check
                

def result_frames(video):
    frame_list, _ = run_model(video, DRAW_BOX=True)
    frame_cnt = -1
    #### 영상 1회재생 ####
    # for frame in frame_list: 
    #     frame_cnt = frame_cnt + 1
    #     frame = frame_list[frame_cnt]
        #### 영상 1회재생 ####
    
    #### 영상 반복재생 ####
    
    while True: 
        frame_cnt = frame_cnt + 1
        if frame_cnt >= len(frame_list):
            frame_cnt = 0
        frame = frame_list[frame_cnt]
        #### 영상 반복재생 ####
        
        # 영상은 초당 약 30프레임이므로 프레임마다 0.03초 마다 출력(데이터 전송)
        time.sleep(0.03)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def cal_attendance(video):
    _, member_check = run_model(video, DRAW_BOX=False)
    return member_check

#디텍션 처리
@app.route('/upload_process', methods=['POST'])
def upload_process():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join('static/image/', filename))
    print(filename,8969)
    y = cal_attendance(f'static/image/{filename}')
    return render_template("result.html",filename=filename,y=y)

@app.route('/result_final')
def result_final():
    filename = request.args.get('filename')
    print('error',filename,'5457')
    return Response(result_frames(f'static/image/{filename}'),mimetype='multipart/x-mixed-replace;boundary=frame')
    # return render_template("result.html", file_name = "image/park.mp4")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/login')
def login():
    return render_template("login.html")

if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)