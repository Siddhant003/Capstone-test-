from flask import Flask,render_template,Response,request
from model_for_gesture import *
from create_gesture_data import *
from trainCNN import *
import cv2
import keras
import threading
import time

flag=0
app=Flask(__name__)
camera=cv2.VideoCapture(0)
word=""
model=""
sp=0
word_dict = {0:'I',1:'love',2:'food',3:'hot'}

def generate_frames():

    global flag,num_frames,num_imgs_taken,model,sp

    model = keras.models.load_model(r"C:\Capstone_dev_ops\code\best_model_3.h5")
    num_frames =0


    while True:
        ## read the camera frame
        success,frame1=camera.read()
        if flag==0:
            frame= fun(frame1,model,num_frames,word_dict)
            num_frames=num_frames+1
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            if sp==1:
                if not t1.is_alive():
                    model=keras.models.load_model(r"C:\Capstone_dev_ops\best_model_3.h5")
                    sp=0
                    print("############","NEW MODEL IN USE","###############")

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            frame=fun1(frame1,word,num_frames,num_imgs_taken)
            num_frames=num_frames+1
            num_imgs_taken=num_imgs_taken+1
            if frame=="done":
                flag=0
                num_frames=0
                num_imgs_taken=0
                t1 = threading.Thread(target=fun2, args=(word_dict,))
                print(word_dict)
                t1.start()
                time.sleep(2)
                sp=1
                continue
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    print(word_dict)
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods =["GET", "POST"])
def rec():
    global flag,word,word_dict,num_frames,num_imgs_taken
    if request.method == "POST":
        flag=1
        word=request.form.get("wname")
        x=max(list(word_dict.keys()))
        x=x+1
        word_dict.update({x:word})
        print(word)
        num_frames=0
        num_imgs_taken=0
    return ('', 204)


if __name__=="__main__":
    app.run(debug=True)
