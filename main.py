from Core import Core
from detection_models import *
import argparse
from Commons import ModelsStrings, Display, ObjectDetectionType, dir_path, Skipping
from logger import Logger

def run_yolo(detection_type, model_path = ""):
    if detection_type is ObjectDetectionType.apple:
        base = "D:\\materialy_pwr\\7sem\\app\\assets\\yolo_tomato_own\\"
        classes = base + "coco_apple.names"
        config = base + 'yolov3_apple_owndataset.cfg'
        if model_path == "":
            weight = base + 'yolov3_apple_owndataset_last.weights'
        else:
            weight = model_path
    elif detection_type is ObjectDetectionType.tomato:
        base = "D:\\materialy_pwr\\7sem\\app\\assets\\yolo_tomato_own\\"
        classes = base + "coco.names"
        config = base + 'yolov3_tomato_owndataset.cfg'
        if model_path == "":
            weight = base + 'yolov3_tomato_RGB_first_last.weights'
        else:
            weight = model_path

    model = YoloV3Module(classes, config, weight)
    return model

def run_maskrcnn(detection_type, model_path = ""):
    if model_path == "":
        if detection_type is ObjectDetectionType.apple:
            model_path = 'D:\\materialy_pwr\\7sem\\app\\assets\\mask_rcnn_tomato_own\\mask_rcnn_applergb_res101_200_aug_153040_0040.h5'
        elif detection_type is ObjectDetectionType.tomato:
            model_path = 'D:\\materialy_pwr\\7sem\\app\\assets\\mask_rcnn_tomato_own\\mask_tomato_best.h5'
            #model_path = 'D:\\materialy_pwr\\7sem\\app\\assets\\mask_rcnn_tomato_own\\mask_rcnn_tomato_0040.h5'

    model = MaskRCNNModule(path_to_wages=model_path, detection_type=detection_type)
    return model

def run_ssd(detection_type, model_path = ""):


    model = SSDModule(os.path.join("D:\\materialy_pwr\\7sem\\app\\assets\\ssd_tomato_own", "ssd_own_dataset1.pb"), detection_type=detection_type)
    #model = SSDModule(os.path.join("D:\\materialy_pwr\\7sem\\app\\assets\\ssd_tomato_own", "apple_best_result.pb"))
    return model

def run_extended_ssd(detection_type, model_path = ""):
    model = run_maskrcnn(ObjectDetectionType.tomato)
    core = Core(model)
    core.run_image("D:\\materialy_pwr\\7sem\\tomato_own_dataset\\dataset_coco\\train\\" + "tomato_2021_08_06_11_04_07046309.png", plot = False)
    model = SSDModule(os.path.join("D:\\materialy_pwr\\7sem\\app\\assets\\ssd_tomato_own", "ssd_own_dataset1.pb"), detection_type=detection_type)
    #model = SSDModule(os.path.join("D:\\materialy_pwr\\7sem\\app\\assets\\ssd_tomato_own", "apple_best_result.pb"))
    return model



def main():
    # format   mask rcnn + output
    # format [xLD, yLD, width, height] yolo
    # format [xymin, xmin, ymax, xmax]  ssd - normalizowane
    # format [start_point, end_point] cv rectangle:

    parser = argparse.ArgumentParser(prog='Test app', description='Object detection test app')

    parser.add_argument('model', metavar='model', type=ModelsStrings, choices=list(ModelsStrings), help='model to execute [mask-rcnn, yolo, ssd]')
    parser.add_argument('display', metavar='display', type=Display, nargs='?', choices=list(Display), help='display type', default=Display.opencv)
    parser.add_argument('-skipping', nargs='?', default=Skipping.lost_frames, const=Skipping.thread, type=Skipping, choices=list(Skipping), help='Skip frames when detecion time is longer')
    parser.add_argument('-model_path', nargs='?', default="")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--apple", action="store_true")
    parser.add_argument('--path', type=dir_path, default="D:\\materialy_pwr\\7sem\\tomato_own_dataset\\video\\fifth.bag")

    args = parser.parse_args()
    detection_type = ObjectDetectionType.tomato
    path = ''
    model_path = ""
    display=Display.plt
    if args.display:
        display = args.display

    if args.model_path:
        model_path = args.model_path

    if args.path:
        path = args.path

    if args.apple:
        detection_type = ObjectDetectionType.apple

    model_choice = args.model

    model = []
    if model_choice is ModelsStrings.mask_rcnn:
        model = run_maskrcnn(detection_type=detection_type)
    elif model_choice is ModelsStrings.yolo:
        model = run_yolo(detection_type=detection_type)
    elif model_choice is ModelsStrings.ssd:
        if args.video:
            model = run_ssd(detection_type=detection_type)
        else:
            model = run_extended_ssd(detection_type=detection_type)
    else:
        raise Exception("Wrong model choice")

    core = Core(model)

    test_video_path = ["D:\\materialy_pwr\\7sem\\tomato_own_dataset\\apple\\video\\video2.bag",
                       "D:\\materialy_pwr\\7sem\\tomato_own_dataset\\video\\fifth.bag"]

    if args.video:
        if model_choice is ModelsStrings.ssd:
            core.run_fast_ssd_video(path, analyze_depth=args.depth, display=display)
        else:
            core.run_video(path, analyze_depth=args.depth, skip_frames=args.skipping, display=display)
    elif args.image:
        path = args.path
        if path == "":
            path = "D:\\materialy_pwr\\7sem\\tomato_own_dataset\\dataset_coco\\train\\" + "tomato_2021_08_06_11_04_07046309.png"
        core.run_image(path)
    elif args.live:
        core.run_live(analyze_depth=args.depth, display=display)


if __name__ == "__main__":
    main()
    sys.stdout.close()