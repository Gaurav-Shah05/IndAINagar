from PIL import Image
import numpy as np
import cv2
import os

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def video2images(video_path, image_path, size, resize=False):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, size)
        cv2.imwrite(image_path + str(i).zfill(10) + '.jpg', frame)
        i += 1
    cap.release()

def images2video(image_path, video_path):
    images = [img for img in os.listdir(image_path)]
    images.sort()
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(video_path, fourcc, frameSize=(width, height), fps=30)
    for image in images:
        frame = cv2.imread(os.path.join(image_path, image))
        out.write(frame)
    out.release()

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def create_heatmap(results, track_ids, track_history, last_positions, frame, heatmap):
    try:
        boxes = results[0].boxes.xywh.cpu()

        for box, track_id in zip(boxes, track_ids):
            x_center, y_center, width, height = box
            current_position = (float(x_center), float(y_center))

            top_left_x = int(x_center - width / 2)
            top_left_y = int(y_center - height / 2)
            bottom_right_x = int(x_center + width / 2)
            bottom_right_y = int(y_center + height / 2)

            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            bottom_right_x = min(heatmap.shape[1], bottom_right_x)
            bottom_right_y = min(heatmap.shape[0], bottom_right_y)

            track = track_history[track_id]
            track.append(current_position)
            if len(track) > 1200:
                track.pop(0)

            last_position = last_positions.get(track_id)
            if last_position and calculate_distance(last_position, current_position) > 5:
                heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

            last_positions[track_id] = current_position
    except:
        pass

    finally:
        heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)

        heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        alpha = 0.7
        if frame.shape != heatmap_color.shape:
            heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

    return overlay, heatmap, track_history, last_positions
       
def resize_frame_to_360p(frame):
    # Convert the OpenCV frame (BGR) to a PIL image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Resize the image to 360p while maintaining aspect ratio
    image.thumbnail((640, 360))
    # Convert the PIL image back to OpenCV format (BGR)
    resized_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return resized_frame
