import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
from argparse import ArgumentParser
from pathlib import Path
import shutil
import pandas as pd
import uuid

from configuration.config import cnn_model_path,yolo_model, yolo_model_path, val_transform, input_videos_dir, output_videos_dir, detect_gap, debug_dir ,debug_crop_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn_model(model_path = cnn_model_path):
    # load cnn name tag detection model
    cnn = resnet18(weights=None)
    cnn.fc = nn.Linear(cnn.fc.in_features, 2)  # binary (0 = no name tag, 1 = name tag)

    cnn.load_state_dict(torch.load(model_path, map_location=device))

    cnn.to(device)
    cnn.eval()

    transform = val_transform

    class_map = {0: "no name tag", 1: "name tag"}

    return cnn, transform, class_map


def load_yolo_model():
    if not yolo_model_path.exists():
        yolo = YOLO(f"{yolo_model}.pt")  # download the model if not present
        shutil.copy(yolo.ckpt_path, yolo_model_path)  # copy to the desired path
    else:
        # load yolo model
        yolo = YOLO(yolo_model_path)
    return yolo


def process_video(video_path, cnn, transform, class_map, yolo, debug_mode=False):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_videos_path = output_videos_dir / f"{video_path.stem}_processed_{cnn_model_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(output_videos_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height)
    )
    
    track_cache = {}  # cache for tracking results
    
    frame_id = 0
    
    record = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1

        # detect and track persons in the frame using YOLO
        results = yolo.track(
            frame,
            classes=[0],  # class 0 is person
            conf=0.15,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        persons = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                track_id = int(box.id.item()) if box.id is not None else None
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # limit the bounding box to the frame dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # skip invalid bounding boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                person_crop = frame[y1:y2, x1:x2]
                
                empty = False
                too_small = False
                
                # skip if the crop is empty
                if person_crop.size == 0:
                    empty = True
                
                # skip if the cropped image is too small
                if person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                    too_small = True
                
                # save the cropped image for debugging
                if debug_mode:
                    cv2.imwrite(str(debug_crop_dir / f"frame_{frame_id}_id_{track_id}_{empty}_{too_small}.jpg"), person_crop)
                
                if empty or too_small:
                    continue

                # preprocess the cropped image for CNN
                pil_image = Image.fromarray(
                    cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                )
                input_tensor = transform(pil_image).to(device)
                persons.append((track_id, (x1, y1, x2, y2), input_tensor))

        persons_batch_map = {}
        batch = []
        for i, (person_id, bbx, input_tensor) in enumerate(persons):
            if person_id is None or person_id not in track_cache or (track_cache[person_id]["last_detect"] + detect_gap <= frame_id):  # if not in cache or it's time to re-detect
                persons_batch_map[i] = len(batch)
                batch.append(input_tensor)
        
        if len(batch) > 0:
            batch = torch.stack(batch).to(device)
            
            # predict using CNN
            with torch.no_grad():
                outputs = cnn(batch)
        
            probs = torch.softmax(outputs, dim=1)
        
        for i, (person_id, bbx, _) in enumerate(persons):
            if i in persons_batch_map:
                prob = probs[persons_batch_map[i]]
                score = prob.max().item()
                pred = prob.argmax().item()
                
                if person_id is not None:
                    # update the cache
                    track_cache[person_id] = {
                        "score": score, 
                        "pred": pred, 
                        "last_seen": frame_id, 
                        "last_detect": frame_id
                    }
            else:
                score = track_cache[person_id]["score"]
                pred = track_cache[person_id]["pred"]
                track_cache[person_id]["last_seen"] = frame_id  # update last seen frame id

            label = f"ID:{person_id if person_id is not None else 'Unknown'} {class_map[pred]} ({score:.2f})"
            
            x1, y1, x2, y2 = bbx
            
            record.append({
                "frame_id": frame_id,
                "person_id": person_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "prediction": pred,
                "score": score
            })

            # draw bounding box and label on the original frame
            if pred == 1:
                color = (0, 255, 0)  # green for name tag
            else:
                color = (0, 0, 255)  # red for no name tag
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        writer.write(frame)
        
        cv2.imshow("Result", frame)
        
        # MAX_INACTIVE_FRAMES = int(fps * 3)  # 3 seconds of inactivity
        
        # for person_id in list(track_cache.keys()):
        #     if frame_id - track_cache[person_id]["last_seen"] > MAX_INACTIVE_FRAMES:  # if the person hasn't been seen for a while
        #         # remove from cache if not seen for a while
        #         del track_cache[person_id]
        
        if cv2.waitKey(1) == 27:  # press 'Esc' to exit
            break
    
    pd.DataFrame(record).to_csv(output_videos_dir / f"{output_videos_path.stem}_record.csv", index=False)
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to the input video file.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing videos.")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode to save cropped images.")
    
    args = parser.parse_args()
    
    debug_mode = args.debug
    
    cnn, transform, class_map = load_cnn_model()
    yolo = load_yolo_model()

    if args.video:
        video_path = Path(args.video)
        process_video(video_path, cnn, transform, class_map, yolo, debug_mode=debug_mode)

    if args.input_dir:
        videos_dir = Path(args.input_dir)

        for video_file in videos_dir.glob("*.mp4"):
            print(f"Processing video: {video_file.name}")
            process_video(video_file, cnn, transform, class_map, yolo, debug_mode=debug_mode)
            print(f"Processed video saved to: {output_videos_dir / video_file.name}")
    
    if not args.video and not args.input_dir:
        videos_dir = input_videos_dir
        
        for video_file in videos_dir.glob("*.mp4"):
            print(f"Processing video: {video_file.name}")
            process_video(video_file, cnn, transform, class_map, yolo, debug_mode=debug_mode)
            print(f"Processed video saved to: {output_videos_dir / video_file.name}")
            
if __name__ == "__main__":
    main()