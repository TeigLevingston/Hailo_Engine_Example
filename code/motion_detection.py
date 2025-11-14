import os
import time
import cv2
import numpy as np
import atexit
import hailo_platform as hpf  # Hailo SDK 4.20 Python bindings

HEF_PATH = "yolov11m.hef"
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# --- Local video file config ------------------------------------------------
# Provide path to an MP4 (or AVI, MKV) file to process. It is a pretty simple change to use a live RTSP Stream
VIDEO_PATH = "<- change to your file"  # 
# ---------------------------------------------------------------------------

# Pretty boilerplate for using OpenCV to grab a stream and frames.
def open_video_capture(path: str):
    cap = cv2.VideoCapture(path)
    return cap, cap.isOpened()

def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame
	
# Just a function to make sure the clips I pull from the stream based on the motion detection contours is 640x640, which is what the model expects
def pad_image(img):
    """Pad or resize an image to 640x640 with black borders, preserving BGR order."""
    target = 640
    h, w = img.shape[:2]
    top = max((target - h) // 2, 0)
    bottom = max(target - h - top, 0)
    left = max((target - w) // 2, 0)
    right = max(target - w - left, 0)
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padded = img
    if padded.shape[0] != target or padded.shape[1] != target:
        padded = cv2.resize(padded, (target, target), interpolation=cv2.INTER_NEAREST)
    return padded
	
# This function returns the detection data in a format I use later.
def parse_hailo_yolo_output(raw_out, frame_shape, conf_th=0.60):
    """
    Normalize Hailo YOLO postprocess outputs into:
      {class_id, class_name, score, bbox=[x1,y1,x2,y2]} 

    """

    H_img, W_img = frame_shape[:2]

    # Raw_Out is an array, [0] is the first element and is a list the 80 COCO classes. I could have done this before the call in line 199
    # Note that GPT-5 couldn't figure this out. 
    # Instead it wrote a bunch of error avoidance/type checking code that caused this to just exit gracefully.
    # I think this is easier.
    out = raw_out[0] 

    for a in out:
        dets = []
        for cls_id, bucket in enumerate(out):
            if bucket is None:
                continue
            arr = bucket
            # normalize to (5, K_i)
            # the shape of the array is determined by the model output, 
            # for this model the output is transposed from how later methods are expecting it.
            # I probably don't need to check since I know the dimensions of the output.
            if hasattr(bucket, "ndim") and bucket.ndim == 2:

                aT = bucket.T
                y1, x1, y2, x2, score = aT

            else:
                # empty or non-array
                continue

            keep = (score >= conf_th) & (x2 > x1) & (y2 > y1)
            if not np.any(keep):
                continue

            y1 = y1[keep]; x1 = x1[keep]; y2 = y2[keep]; x2 = x2[keep]; score = score[keep]

            # Hailo YOLO outputs normalized boxes in [0,1] range according to their documentation. The if block makes sure that is the case and adjusts from normalized to pixels
            if max(float(np.max(x2, initial=0)), float(np.max(y2, initial=0))) <= 1.5:
                x1 = (x1 * W_img).astype(int); x2 = (x2 * W_img).astype(int)
                y1 = (y1 * H_img).astype(int); y2 = (y2 * H_img).astype(int)
            else:
                x1 = x1.astype(int); x2 = x2.astype(int); y1 = y1.astype(int); y2 = y2.astype(int)


            dets.append({
				"class_id": int(cls_id),
				"class_name": COCO80[cls_id] if 0 <= cls_id < len(COCO80) else str(cls_id),
				"score": float(score),
				"bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        return dets
    
# This function sopports saving the bounding box in YOLO format. I use that in YOLOLabel, which is a great tool for creating training data.
# I will be trainng a model later with the data I collect.    
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

# I need this because of how I am doing motion detection and cropping. Most examples implement this a local
# and then send a stream to it. That lets the model load once and works. I need the persistant instance
# So can pass multiple frames to it without getting the hailo_platform.pyhailort.pyhailort.hailortstatus exception: 8
# This is based on Hailo SDK examples and my own experiments.
# My project is to compare the accuracy and performance of the model between using a high-resolution frame that has been resized as input, the way most stream examples work, 
# and a  frame that was cropped, padded, and only, sometimes, resized.
#If you are interested in Hailo SDK programming, I hope this is helpful.
class HailoInferenceEngine:
    """Persistent Hailo YOLO inference engine.
    Loads HEF, configures device and opens InferVStreams once. Call infer(frame) to get detections.
    This is loosely based on Hailo SDK examples, especially the common\hailo_inference.py example,
    combined with work I had done with some other examples. This approach lets you call multiple inferences against a model without reloading it.

        """
    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self.hef = hpf.HEF(hef_path)
        #This creates the virtual device and keeps it open. We use that to load the model and set the parameters.
        self.vdev = hpf.VDevice() 
        # This reads the HEF and configures the device. If you are using a USB device, change interface to USB.
        # You can run hailortcli on the hef to see some of the values it expects from some parameters.
        # As long as you have a valid hef, the process is to create the vdevice, assign it to an interface, load
        # the network group information, and activate it.
        cfg = hpf.ConfigureParams.create_from_hef(self.hef, interface=hpf.HailoStreamInterface.PCIe)
        # Now that the vdevice is created and configured, we can load the model onto it. The call returns a list and we need the first one.
        #This may be important later if you want to load multiple models.
        self.network_group = self.vdev.configure(self.hef, cfg)[0]
        #This stores the network group and in/outparameters, we need them to activate the network later.
        self.ng_params = self.network_group.create_params()
        in_info = self.hef.get_input_vstream_infos()[0]
        out_info = self.hef.get_output_vstream_infos()[0]
        self.in_name, self.out_name = in_info.name, out_info.name
        self.in_shape = tuple(in_info.shape)  # (H,W,C) for NHWC
        #This was confusing at first since the N value is implicitly 1 for batch size. We add that dimension later.
        H, W, C = self.in_shape
        # All of the above is pretty boilerplate and you can find other examples of the approach. 
		# We are reading parameters from the model and saving them to use during activation.
        
        # This just creates the input and output Vstream parameters based on the network group. 
        # Quantized is true because YOLO11M allows it, this model works either way on my Hailo8L.
        # The format types are based on the model. See the hailortcli output. I will be trying to see if that affects performance.
        self.in_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )
        self.out_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.FLOAT32
        )
        # This creates the activation context and activates them. If you debug and break through thes steps
        # you can see the resources being allocated on the device as local variables.

        self.activation = self.network_group.activate(self.ng_params)
        self.activation.__enter__()
        self.infer_pipe = hpf.InferVStreams(self.network_group, self.in_params, self.out_params)
        self.infer_pipe.__enter__()

        #In the main code this is the method that you call with an image as a parameter to get detections.
        #The conf_th parameter allows you to set the confidence threshold for detections.

    def infer(self, frame_bgr: np.ndarray, conf_th: float = 0.60):

        if frame_bgr is None or frame_bgr.size == 0:
            return []
        H, W, C = self.in_shape
        # Resize if needed
        if frame_bgr.shape[0] != H or frame_bgr.shape[1] != W:
            frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        # The model expects RGB input (see hailorrtcli output) and it was created with cv2.videocapture which is BGR. We convert from BGR to RGB.
		# If the colors are messed up you would skip this.
        
        inp_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
        #This is where we add the missing dimension for batch size (the missing N in NHWC from above)
        inp = np.expand_dims(inp_rgb, axis=0)
        inp = np.ascontiguousarray(inp)
    

		# This just double checks that the input frame is the right shape.
        expected_shape = (1, H, W, C)
        if inp.shape != expected_shape:
            print(f"Bad input shape {inp.shape}, expected {expected_shape}")
            return []
        results = self.infer_pipe.infer({self.in_name: inp})
        raw_out = results[self.out_name]
        return parse_hailo_yolo_output(raw_out, frame_bgr.shape, conf_th=conf_th)

    def close(self):
        # Properly release contexts
        if hasattr(self, 'infer_pipe') and self.infer_pipe:
            self.infer_pipe.__exit__(None, None, None)
        if hasattr(self, 'activation') and self.activation:
            self.activation.__exit__(None, None, None)
        if hasattr(self, 'vdev') and self.vdev:
            self.vdev.release()


engine: HailoInferenceEngine | None = None

def init_engine():
    global engine
    if engine is None:
        engine = HailoInferenceEngine(HEF_PATH)
        atexit.register(engine.close)
    return engine

def run_inference(img):
    eng = init_engine()
    return eng.infer(img)

                        

# From here down is the motion detection and main loop code. I am pulling frames from an rtsp stream, 
# Using OpenCV for motion detection and cropping. 
# Then passing the cropped frames to the HailoInferenceEngine instance for detection.
# This lets me keep the model loaded once and avoid the hailortstatus exception: 8


def detect_motion():
    # Open local video file once
    cap, opened = open_video_capture(VIDEO_PATH)
    if not opened:
        raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")
    try:
        while True:
            frame1 = read_frame(cap)
            if frame1 is None:
                # End of file
                print("End of video (frame1).")
                break


            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame1 = cv2.GaussianBlur(gray_frame1, (21, 21), 0)
            # Short delay to simulate motion window; for file we can skip or use small sleep
            # time.sleep(0.05)
            frame2 = read_frame(cap)
            if frame2 is None:
                print("End of video (frame2).")
                break

            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.GaussianBlur(gray_frame2, (21, 21), 0)

            # Calculate the absolute difference between the current frame and the first frame
            frame_delta = cv2.absdiff(gray_frame1, gray_frame2)

            # Threshold the difference image to highlight areas of change
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            # Dilate the thresholded image to fill in gaps and make contours more visible
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            counter = 0
            os.makedirs("images", exist_ok=True)
            for contour in contours:
                # Skip small contours (noise)
                print("Contours " + str(len(contours)))                
                if cv2.contourArea(contour) <= 500:

                    continue

                # Bounding box around the moving region
                x, y, w, h = cv2.boundingRect(contour)
                # Work on a copy to avoid any aliasing issues later in the loop
                frame_for_crop = frame2.copy()

                if w >= 640 and h >= 640:
                    # Large motion region: take the rect and later pad/resize to 640
                    x1 = max(x, 0)
                    y1 = max(y, 0)
                    x2 = min(x + w, frame_for_crop.shape[1])
                    y2 = min(y + h, frame_for_crop.shape[0])
                else:
                    # Centered 640x640 crop around the motion centroid
                    cx = x + w // 2
                    cy = y + h // 2
                    half = 320
                    x1 = max(cx - half, 0)
                    y1 = max(cy - half, 0)
                    x2 = min(cx + half, frame_for_crop.shape[1])
                    y2 = min(cy + half, frame_for_crop.shape[0])

                cropped_motion = frame_for_crop[y1:y2, x1:x2]
                if cropped_motion is None or cropped_motion.size == 0:
                    continue
                if cropped_motion.shape[0] != 640 or cropped_motion.shape[1] != 640:
                    cropped_motion = pad_image(cropped_motion)

                # Run inference for this contour
                print(counter)                
                detections = run_inference(cropped_motion)
                motion_detected = True
                counter += 1

                # Save results per contour if any detections
                if detections:
                    yolo_lines = []
                    h_img, w_img = cropped_motion.shape[:2]
                    for d in detections:
                        bbox = d.get('bbox')
                        cid = d.get('class_id')
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            xc, yc, ww, hh = pascal_voc_to_yolo(x1, y1, x2, y2, w_img, h_img)
                            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    base = f"images/{ts}_{counter:03d}"
                    cv2.imwrite(base + "_motion.jpg", cropped_motion)
                    with open(base + "_motion.txt", "w") as f:
                        for line in yolo_lines:
                            f.write(line + "\n")

                print("Motion detected " + str(len(detections)))

            if motion_detected:
                continue
        
    except Exception as e:
        print(f"Error in detect_motion loop: {e}")
    finally:
        try:
            cap.release()
        except Exception:
            pass
                #cv2.putText(frame1, "Motion Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.imwrite(framename + "_Frame.png", frame1) 
                #cv2.imwrite(framename + "_Thresh.png", thresh)


def main():
    #infer_pipe=open_hailo_device(HEF_PATH)

    detect_motion()
       
if __name__ == "__main__":
    main()
