import cv2
import glob
import time

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

use_cuda = True
num_classes = 80

namesfile = ROOT_DIR + '/data/coco.names'
weightfile = ROOT_DIR + '/data/yolov4.weights'
cfgfile = ROOT_DIR + '/cfg/yolov4-custom.cfg'

m = Darknet(cfgfile)
m.load_weights(weightfile)

class_names = load_class_names(namesfile)


class HEIGHTS_COLLECTION:
    # magic :)
    focal_length = 320
    classes = {
        'person': 1.75,
        'car': 1.45,
        'bus': 3.0,
        'truck': 3.0
    }


def plot_distance(img, boxes):
    width = img.shape[1]
    height = img.shape[0]

    mask = np.zeros((height, width, 1), np.uint8)

    for i in range(len(boxes)):
        box = boxes[i]

        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        class_name = class_names[box[6]]

        if class_name in HEIGHTS_COLLECTION.classes:
            distance = HEIGHTS_COLLECTION.focal_length * (HEIGHTS_COLLECTION.classes[class_name] / (y2 - y1))
            distance = round(distance, 2)

            brightness = (30 - distance) * 8.5

            cv2.rectangle(mask, (x1, y1), (x2, y2), brightness, cv2.FILLED)
            cv2.putText(img, "distance: " + str(distance) + " m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 24, 158), 2)

    return img, mask


if use_cuda:
    m.cuda()

videos = glob.glob(ROOT_DIR + '/videos/*.mp4')

for fname in videos:
    video_capture = cv2.VideoCapture(fname)
    print('Video', fname, 'start detection.')

    start_time = time.time()
    x = 1
    counter = 0
    FPS = 0

    while video_capture.isOpened():
        t0 = time.time()

        success, frame = video_capture.read()
        if not success:
            break

        t1 = time.time()

        # print('Frame reading: %f' % (t1 - t0))

        sized = cv2.resize(frame, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        t2 = time.time()

        # print('Image transformation: %f' % (t2 - t1))

        boxes = do_detect(m, sized, 0.5, num_classes, 0.4, use_cuda)

        t3 = time.time()

        # print('Detection: %f' % (t3 - t2))

        frame = plot_boxes_cv2(frame, boxes, savename=None, class_names=class_names)
        frame, mask = plot_distance(frame, boxes)

        t4 = time.time()

        # print('Debugging: %f' % (t4 - t3))

        counter += 1
        if (time.time() - start_time) > x:
            FPS = round(counter / (time.time() - start_time), 2)
            print('FPS:', FPS)

            counter = 0
            start_time = time.time()

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        t5 = time.time()

        """
        print('Perfomance: %f' % (t5 - t4))
        print('TOTAL: %f' % (t5 - t0))
        print('--------------------------')
        """

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
