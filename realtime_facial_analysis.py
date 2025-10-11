import cv2
import mediapipe as mp
import math
import numpy as np
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from threading import Thread
from collections import deque

warnings.simplefilter("ignore", UserWarning)

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src)
        
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()
                
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()
        
    def isOpened(self):
        return not self.stopped

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion, eps=0.001, momentum=0.99)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        
        return x

class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv_layer_s2_same = Conv2dSame(num_channels, 64, 7, stride=2, groups=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*ResBlock.expansion, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        x = self.relu(self.batch_norm1(self.conv_layer_s2_same(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride, bias=False, padding=0),
                nn.BatchNorm2d(planes*ResBlock.expansion, eps=0.001, momentum=0.99)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

class LSTMPyTorch(nn.Module):
    def __init__(self):
        super(LSTMPyTorch, self).__init__()
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)        
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x

class RealtimeFacialAnalyzer:
    def __init__(self, model_path, device='cuda', lstm_model_path=None, use_temporal_smoothing=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.emotions = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
        
        self.model = ResNet50(num_classes=7, channels=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.half()
            torch.backends.cudnn.benchmark = True
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.lstm_model = None
        if lstm_model_path and Path(lstm_model_path).exists():
            self.lstm_model = LSTMPyTorch()
            self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
            self.lstm_model.to(self.device)
            self.lstm_model.eval()
        
        self.use_temporal_smoothing = use_temporal_smoothing
        self.lstm_features = []
        self.prediction_history = []
        self.history_size = 10
        
        self.prev_box = None
        self.box_smooth_factor = 0.3
        self.min_face_size = 40
        
        self.face_detect_interval = 2
        self.inference_interval = 1
        self.last_emotion = "Neutral"
        self.last_confidence = 0.0
        
        self.detection_scale = 0.5
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self._warmup_model()
    
    def _warmup_model(self):
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        if torch.cuda.is_available():
            dummy_input = dummy_input.half()
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def preprocess_face(self, face_img):
        face_resized = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_NEAREST)
        face_float = face_resized.astype(np.float32)
        face_float = face_float[..., ::-1].copy()
        face_float[..., 0] -= 91.4953
        face_float[..., 1] -= 103.8827
        face_float[..., 2] -= 131.0912
        face_tensor = torch.from_numpy(face_float).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        if torch.cuda.is_available():
            face_tensor = face_tensor.half()
        return face_tensor
    
    def norm_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def smooth_box(self, new_box):
        if self.prev_box is None:
            self.prev_box = new_box
            return new_box
        
        smoothed = []
        for i in range(4):
            smoothed.append(int(self.prev_box[i] * (1 - self.box_smooth_factor) + new_box[i] * self.box_smooth_factor))
        
        self.prev_box = tuple(smoothed)
        return tuple(smoothed)
    
    def is_valid_face(self, startX, startY, endX, endY):
        width = endX - startX
        height = endY - startY
        
        if width < self.min_face_size or height < self.min_face_size:
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        return True
    
    def get_box(self, fl, w, h):
        idx_to_coors = {}
        for idx, landmark in enumerate(fl.landmark):
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)
            if landmark_px:
                idx_to_coors[idx] = landmark_px

        x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
        y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
        endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
        endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])

        startX, startY = max(0, x_min), max(0, y_min)
        endX, endY = min(w - 1, endX), min(h - 1, endY)
        
        if not self.is_valid_face(startX, startY, endX, endY):
            return None
        
        return self.smooth_box((startX, startY, endX, endY))

    def display_emotion(self, img, box, label='', line_width=2):
        lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
        color = (255, 0, 255)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        tf = max(lw - 1, 1)
        text_fond = (0, 0, 0)
        text_width, _ = cv2.getTextSize(label, font, lw / 3, tf)
        text_width = text_width[0] + round(((p2[0] - p1[0]) * 10) / 360)
        center_face = p1[0] + round((p2[0] - p1[0]) / 2)

        cv2.putText(img, label, (center_face - round(text_width / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), 
                    font, lw / 3, text_fond, thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (center_face - round(text_width / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), 
                    font, lw / 3, color, thickness=tf, lineType=cv2.LINE_AA)
        return img

    def display_fps(self, img, text, margin=1.0, box_scale=1.0):
        img_h, img_w, _ = img.shape
        line_width = int(min(img_h, img_w) * 0.001)
        thickness = max(int(line_width / 3), 1)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        font_scale = thickness / 1.5

        t_w, t_h = cv2.getTextSize(text, font_face, font_scale, None)[0]
        margin_n = int(t_h * margin)
        sub_img = img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
                  img_w - t_w - margin_n - int(2 * t_h * box_scale): img_w - margin_n]

        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
        img_w - t_w - margin_n - int(2 * t_h * box_scale):img_w - margin_n] = cv2.addWeighted(sub_img, 0.5, white_rect, .5, 1.0)

        cv2.putText(img=img, text=text,
                    org=(img_w - t_w - margin_n - int(2 * t_h * box_scale) // 2,
                         0 + margin_n + t_h + int(2 * t_h * box_scale) // 2),
                    fontFace=font_face, fontScale=font_scale, color=font_color,
                    thickness=thickness, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        return img
    
    def predict_emotion(self, face_img):
        face_tensor = self.preprocess_face(face_img)
        
        with torch.inference_mode():
            if self.lstm_model:
                features = torch.nn.functional.relu(self.model.extract_features(face_tensor))
                
                if len(self.lstm_features) == 0:
                    self.lstm_features = [features] * self.history_size
                else:
                    self.lstm_features = self.lstm_features[1:] + [features]
                
                lstm_input = torch.cat(self.lstm_features, dim=0).unsqueeze(0)
                probabilities = self.lstm_model(lstm_input)
                confidence, predicted = torch.max(probabilities, 1)
            else:
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                if self.use_temporal_smoothing:
                    probs_np = probabilities[0].cpu().numpy()
                    self.prediction_history.append(probs_np)
                    if len(self.prediction_history) > self.history_size:
                        self.prediction_history.pop(0)
                    
                    smoothed_probs = np.mean(self.prediction_history, axis=0)
                    predicted = np.argmax(smoothed_probs)
                    confidence_score = smoothed_probs[predicted]
                    emotion = self.emotions[predicted]
                    return emotion, float(confidence_score)
                
                confidence, predicted = torch.max(probabilities, 1)
        
        emotion = self.emotions[predicted.item()]
        confidence_score = confidence.item()
        
        return emotion, confidence_score
    
    def run_webcam(self):
        stream = WebcamStream(src=0).start()
        time.sleep(1.0)
        
        if not stream.isOpened():
            print("Error: Could not open webcam")
            return
        
        frame = stream.read()
        h, w = frame.shape[:2]
        
        print(f"Running on: {self.device}")
        print(f"Resolution: {w}x{h}")
        print(f"Optimizations enabled:")
        print(f"  - Threaded webcam capture")
        print(f"  - Downscaled face detection (50% resolution)")
        print(f"  - Face detect every {self.face_detect_interval} frames")
        print(f"  - Inference every {self.inference_interval} frames")
        if torch.cuda.is_available():
            print(f"  - FP16 half precision")
            print(f"  - CuDNN auto-tuning")
        print("Press 'q' to quit")
        
        frame_count = 0
        last_box = None
        fps_values = deque(maxlen=30)
        
        while stream.isOpened():
            t1 = time.time()
            frame = stream.read()
            
            if frame is None:
                continue
            
            frame_count += 1
            run_detection = (frame_count % self.face_detect_interval == 0)
            run_inference = (frame_count % self.inference_interval == 0)
            
            if run_detection:
                detect_h = int(h * self.detection_scale)
                detect_w = int(w * self.detection_scale)
                frame_small = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)
                
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.face_mesh.process(frame_rgb)
                frame_rgb.flags.writeable = True

                if results.multi_face_landmarks:
                    fl = results.multi_face_landmarks[0]
                    box_coords = self.get_box(fl, detect_w, detect_h)
                    
                    if box_coords is not None:
                        box_coords = tuple(int(coord / self.detection_scale) for coord in box_coords)
                        last_box = box_coords
                        
                        if run_inference:
                            frame_full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            startX, startY, endX, endY = box_coords
                            cur_face = frame_full_rgb[startY:endY, startX:endX]
                            
                            if cur_face.size > 0:
                                try:
                                    self.last_emotion, self.last_confidence = self.predict_emotion(cur_face)
                                except:
                                    pass
                else:
                    if frame_count % (self.face_detect_interval * 3) == 0:
                        last_box = None
                        self.prev_box = None
            
            if last_box is not None:
                label = f"{self.last_emotion} {self.last_confidence*100:.1f}%"
                frame = self.display_emotion(frame, last_box, label, line_width=3)

            t2 = time.time()
            elapsed = t2 - t1
            if elapsed > 0:
                fps = 1 / elapsed
                fps_values.append(fps)
            
            if len(fps_values) > 0:
                avg_fps = np.mean(fps_values)
                frame = self.display_fps(frame, f'FPS: {avg_fps:.1f}', box_scale=.5)
            
            cv2.imshow('Real-time Facial Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        stream.stop()
        cv2.destroyAllWindows()

def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "models" / "FER_static_ResNet50_AffectNet.pt"
    lstm_model_path = script_dir / "models" / "FER_dinamic_LSTM_Aff-Wild2.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Please place the trained model file 'FER_static_ResNet50_AffectNet.pt' in: {script_dir / 'models'}")
        sys.exit(1)
    
    use_lstm = lstm_model_path.exists()
    if use_lstm:
        print(f"LSTM model found - Using temporal analysis")
        analyzer = RealtimeFacialAnalyzer(str(model_path), lstm_model_path=str(lstm_model_path))
    else:
        print(f"LSTM model not found - Using temporal smoothing")
        analyzer = RealtimeFacialAnalyzer(str(model_path), use_temporal_smoothing=True)
    
    analyzer.run_webcam()

if __name__ == "__main__":
    main()

