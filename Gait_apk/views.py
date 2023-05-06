import cv2
from django.shortcuts import render
from .forms import VideoForm
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from keras.models import Sequential, model_from_json, load_model
import glob
import time


def mass_center(img, is_round=True):
    Y = img.mean(axis=1)
    X = img.mean(axis=0)
    Y_ = np.sum(np.arange(Y.shape[0]) * Y)/np.sum(Y)
    X_ = np.sum(np.arange(X.shape[0]) * X)/np.sum(X)
    if is_round:
        return int(round(X_)), int(round(Y_))
    return X_, Y_


def image_extract(img, newsize):
    if (len(np.where(img.mean(axis=0) != 0)[0]) != 0):
        x_s = np.where(img.mean(axis=0) != 0)[0].min()
        x_e = np.where(img.mean(axis=0) != 0)[0].max()

        y_s = np.where(img.mean(axis=1) != 0)[0].min()
        y_e = np.where(img.mean(axis=1) != 0)[0].max()

        x_c, _ = mass_center(img)
        x_s = x_c-newsize[1]//2
        x_e = x_c+newsize[1]//2
        img = img[y_s:y_e, x_s if x_s > 0 else 0:x_e if x_e <
                  img.shape[1] else img.shape[1]]
        return cv2.resize(img, newsize)
    else:
        return 0


lables = {'Abhirami': 0,
          'Aswathy': 1,
          'Ayana': 2,
          'Lekshmi': 3,
          'Nandana': 4,
          'Parthiv': 5,
          'Shilpa': 6, }


def process_video(file):
    # file_content = file.read()
    # print(str(file_content))
    # buffer = BytesIO(file_content)
    # print(buffer.getvalue())
    # Open the video file
    vidcap = cv2.VideoCapture('videos/' + str(file))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps)

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    count = 0
    mask_count = 0
    thresh = 127
    message = 'Analyzing...'
    m2 = ' '
    fps = 0

    pretrained_weight = './yolov8x-seg.pt'
    model = YOLO(pretrained_weight)
    new_model = load_model('./fine_tuned.h5')

    mask_dir = './output/masks/'
    mask_files = glob.glob(mask_dir + '/*')
    for f in mask_files:
        os.remove(f)

    while success:
        start = time.time()
        if (count % 4 == 0):
            results = model(image)
            masks = results[0].masks
            if masks:
                m2 = 'YOLO v8 = Inference: {0} ms | Preprocess: {1} ms | Postprocess: {2} ms' .format(round(
                    results[0].speed['inference'], 2), round(results[0].speed['preprocess'], 2), round(results[0].speed['postprocess'], 2))
                mask_count = mask_count+1
                ms = masks.data.numpy()
                cv2.imwrite(mask_dir + str(count) + '.png', ms[0, :, :]*255)

        if (mask_count >= 10):
            image_data = []
            files = os.listdir('./output/masks/')
            for f in files:
                im = cv2.imread('./output/masks/'+'/'+f, 0)
                im_bw = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
                item = image_extract(im_bw, (64, 128))
                if (np.max(item) != 0):
                    image_data.append(item)
            gei = np.mean(image_data, axis=0)

            res_img = cv2.resize(gei, (224, 224))
            test_img = cv2.merge([res_img, res_img, res_img])
            test_img = test_img/255
            test_img = np.reshape(test_img, (1, 224, 224, 3))

            preds = new_model.predict(test_img)
            print(preds)
            prediction = np.argmax(preds)
            print(prediction)
            for name, label in lables.items():
                p=str(round(preds[0][prediction]*100, 2))
                floatp=float(p)
                #print(floatp)
                if floatp < 90:
                        message ='Unknown Person Detected'
                elif label == prediction:
                    
                    message = 'Detected : ' + name + \
                        ' (' + \
                        str(round(preds[0][prediction]*100, 2)) + '% Accuracy)'

        success, image = vidcap.read()
        count += 1

        mask_out = cv2.merge(
            [ms[0, :, :]*255, ms[0, :, :]*255, ms[0, :, :]*255])

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (25, 35)
        org1 = (25, 60)
        org2 = (25, 330)
        org3 = (495, 330)
        fontScale = 0.8
        fontScale2 = 0.45
        fontScale3 = 0.65
        color = (0, 0, 0)
        color2 = (0, 0, 255)
        color3 = (255, 255, 255)
        color4 = (255, 0, 255)
        color5 = (0, 255, 0)
        thickness = 2
        thickness2 = 1

        title1 = 'Live'
        org4 = (25, 30)
        title2 = 'YOLO v8 generated mask'

        fps_msg = 'FPS: ' + str(round(fps/1000, 2))
        image = cv2.putText(image, fps_msg, org3, font,
                            fontScale3, color2, thickness, cv2.LINE_AA)
        image = cv2.putText(image, title1, org4, font,
                            fontScale3, color2, thickness, cv2.LINE_AA)

        mask_out = cv2.putText(mask_out, title2, org4, font,
                               fontScale3, color4, thickness, cv2.LINE_AA)

        # mask_out = cv2.putText(mask_out, m2, org2, font,
        #                 fontScale2, color5, thickness2, cv2.LINE_AA)

        prompt = np.zeros((95, 640, 3))*255
        prompt = cv2.putText(prompt, message, org, font,
                             fontScale, color5, thickness, cv2.LINE_AA)
        prompt = cv2.putText(prompt, m2, org1, font,
                             fontScale2, color3, thickness2, cv2.LINE_AA)
        if (image is None):
            cv2.destroyAllWindows()
        if (image is not None):
            # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
            disp = np.concatenate(
                (cv2.resize(image/255, (640, 280)), cv2.resize(mask_out, (640, 280))), axis=0)
            info = np.concatenate((disp, prompt), axis=0)
            cv2.imshow('Live', info)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
        end = time.time()
        seconds = end - start
        fps = length / seconds

    # # Release the video file and destroy the window
    # cap.release()
    # cv2.destroyAllWindows()


def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['file']
            print(name)
            form.save()
            # Process the uploaded video
            process_video(name)
            return render(request, 'upload_video.html', {'form': form, 'message': 'Video processed successfully!'})
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def info(request):
    return render(request,'info.html')
    