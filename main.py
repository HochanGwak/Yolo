from fileinput import filename
import re
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
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
import tensorflow.compat.v1 as tf
import pickle
from PIL import Image
from werkzeug.utils import secure_filename


DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'


modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"

global x




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

def result_frames(video):
    # cap = cv2.VideoCapture(video)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # time.sleep(0.2)
    # lastTime = time.time()*1000.0

    # while True:
    #     ret, image = cap.read()
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       

    #     delt = time.time()*1000.0-lastTime
    #     s = str(int(delt))
    #     lastTime = time.time()*1000.0


    #     # cv2.putText(image, s, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #     # now = datetime.datetime.now()
    #     # timeString = now.strftime("%Y-%m-%d %H:%M")
    #     # cv2.putText(image, timeString, (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #     cv2.imshow("Frame", image)
    #     key = cv2.waitKey(1) & 0xFF
    #  # if the `q` key was pressed, break from the loop
    #     if key == ord("q"):
    #         break
   
    #     ret, buffer = cv2.imencode('.jpg', image)
    #     frame = buffer.tobytes()
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    global x
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
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
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')
            
            video_capture = cv2.VideoCapture(video)
            print('Start Recognition')
            a = set(HumanNames)
            tm = time.time()
            while True:
                ret, frame = video_capture.read()
            
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
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
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                                        a = a - set(result_names.split())
                                        
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                        except:   
                            
                            print("error")
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow('Face Recognition', frame)
                print(endtimer-tm,'111')
                if (endtimer - tm) >30:
                    if len(list(a)) == 0:
                        x = '전원출석'
                        print(x)
                    else:
                        print('결석 인원 : ',end='')
                        for i in list(a):
                            print(i,end=' ')


                    break
                # key= cv2.waitKey(1)
                # if key== 113: # "q"
                #     #직접 작성스타트
                #     # print(result_names,'aaa')
                #     if len(list(a)) == 0:
                #         print('전원출석')
                #     else:
                #         print('결석 인원 : ',end='')
                #         for i in list(a):
                #             print(i,end=' ')
                
                #     break
            
                ret, buffer = cv2.imencode('.jpg', frame)
                frame1 = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
    print(x)
    return x   
           






#디텍션 처리
@app.route('/upload_process', methods=['POST'])
def upload_process():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join('static/image/', filename))
    print(filename,8969)
    y = result_frames(f'static/image/{filename}')
    print(y,666)
    # nparr = np.frombuffer(file.read(), np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # filename_first = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    # dst = tflite_catdog.run(image)

    # json_path = os.path.join('outputs', filename_first + '.json')
    # json.dump({'rate':float(dst)},open(json_path,'w'))
    
    # dst_path = os.path.join('outputs', filename_first + '.jpg')
    # cv2.imwrite(dst_path, image)
    
    return redirect(url_for('result',filename=filename,y=y))

# #outputs 폴더를 일반 웹서버 형식으로 오픈
# @app.route('/outputs/<path:filename>', methods=['GET', 'POST'])
# def download(filename):
#     output_path = os.path.join(app.root_path, 'outputs')
#     return send_from_directory('outputs', filename)

#디텍션 결과 보여주기
@app.route('/result')
def result():
    filename = request.args.get('filename')
    y = request.args.get('y')
    print(y,7770)
    # result_frames(f'static/image/{filename}')
    # print(x,890)
    # return Response(result_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')
    # return render_template("result.html", file_name = "image/park.mp4")
    return render_template("result.html",filename=filename,y=y)

@app.route('/result_final')
def result_final():
    filename = request.args.get('filename')
    print('error',filename,'5457')
    return Response(result_frames(f'static/image/{filename}'),mimetype='multipart/x-mixed-replace;boundary=frame')
    # return render_template("result.html", file_name = "image/park.mp4")



# # API 서비스
# @app.route('/object_detection', methods=['POST'])
# def infer():
#     nparr = np.frombuffer(request.data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     rects, classes, scores = tflite_detector.inference(img)

#     result = {'rects':rects.tolist(), 'classes':classes.tolist(), 'scores':scores.tolist()}
#     response = json.dumps(result)

#     return Response(response=response, status=200, mimetype="application/json")

# import io
# # API 서비스
# @app.route('/color_service', methods=['POST'])
# def color_service():
#     nparr = np.frombuffer(request.data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     dst = tflite_color.run(img)

#     _, buf = cv2.imencode('.jpg', dst)
#     return flask.send_file(io.BytesIO(buf), download_name='result.jpg', mimetype='image/jpeg')



# start flask app
def main():
    # os.makedirs('outputs', exist_ok=True)
    # tflite_detector.load_model("detect.tflite")
    # tflite_catdog.load_model('dogcat.tflite')
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
