import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pytesseract
from ultralytics import YOLO
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Path):
            return str(o)
        return super(NumpyEncoder, self).default(o)


class VideoFeatureExtractor:

    def __init__(
        self,
        video_path: str,
        yolo_model_version: str,
        sample_rate: int = 5,
    ):
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.yolo_model_version = yolo_model_version
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.duration = None
        self.yolo_model = YOLO(
            f"{yolo_model_version}.pt"
        )  # using small model for faster process (can use n,m,l,x models as well)

    def _open_video(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = float(self.total_frames / self.fps) if self.fps > 0 else 0.0

        return True

    def _close_video(self):
        if self.cap is not None:
            self.cap.release()

    def detect_shot_cuts(self, threshold: float = 0.5) -> Dict:
        self._open_video()

        prev_hist = None
        cuts = []
        frame_count = 0
        shot_sample_rate = 5

        if self.cap is None:
            self._close_video()
            return {"total_cuts": 0, "cuts": [], "average_shot_length": 0.0}

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % shot_sample_rate != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = np.sum(np.abs(hist - prev_hist))

                # Detect scene changes: typical values 0.5-1.0 for cuts
                if diff > threshold:
                    fps = self.fps if self.fps is not None and self.fps > 0 else 30.0
                    timestamp = frame_count / fps
                    cuts.append(
                        {
                            "frame": frame_count,
                            "timestamp": round(float(timestamp), 2),
                            "difference": round(float(diff), 2),
                        }
                    )

            prev_hist = hist

        self._close_video()

        duration = self.duration if self.duration is not None else 0.0
        avg_shot_length = float(duration / (len(cuts) + 1)) if cuts else float(duration)

        return {
            "total_cuts": len(cuts),
            "cuts": cuts,
            "average_shot_length": round(avg_shot_length, 2),
        }

    def analyze_motion(self) -> Dict:
        self._open_video()

        prev_gray = None
        motion_magnitudes = []
        frame_count = 0

        if self.cap is None:
            self._close_video()
            return {"average_motion": 0.0, "max_motion": 0.0, "min_motion": 0.0}

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % self.sample_rate != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                avg_magnitude = np.mean(magnitude)
                motion_magnitudes.append(float(avg_magnitude))

            prev_gray = gray

        self._close_video()

        if not motion_magnitudes:
            return {"average_motion": 0.0, "max_motion": 0.0, "min_motion": 0.0}

        return {
            "average_motion": round(float(np.mean(motion_magnitudes)), 2),
            "max_motion": round(float(np.max(motion_magnitudes)), 2),
            "min_motion": round(float(np.min(motion_magnitudes)), 2),
            "motion_std": round(float(np.std(motion_magnitudes)), 2),
        }

    def detect_text(self, confidence_threshold: int = 60) -> Dict:
        self._open_video()

        frames_with_text = 0
        total_sampled_frames = 0
        all_words = []
        frame_count = 0

        if self.cap is None:
            self._close_video()
            return {
                "text_present_ratio": 0.0,
                "frames_with_text": 0,
                "total_sampled_frames": 0,
                "top_keywords": [],
                "unique_words": 0,
            }

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % (self.sample_rate * 2) != 0:
                continue

            total_sampled_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                data = pytesseract.image_to_data(
                    binary, output_type=pytesseract.Output.DICT
                )

                frame_has_text = False
                for i, conf in enumerate(data["conf"]):
                    if int(conf) > confidence_threshold:
                        text = data["text"][i].strip()
                        if text and len(text) > 2:
                            frame_has_text = True
                            all_words.append(text.lower())

                if frame_has_text:
                    frames_with_text += 1
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

        self._close_video()

        text_present_ratio = (
            frames_with_text / total_sampled_frames if total_sampled_frames > 0 else 0.0
        )

        word_counts = Counter(all_words)
        top_keywords = [word for word, count in word_counts.most_common(10)]

        return {
            "text_present_ratio": round(float(text_present_ratio), 3),
            "frames_with_text": frames_with_text,
            "total_sampled_frames": total_sampled_frames,
            "top_keywords": top_keywords,
            "unique_words": len(word_counts),
        }

    def detect_objects_and_people(self) -> Dict:
        self._open_video()

        person_count = 0
        object_count = 0
        frame_count = 0
        person_id = 0
        detections_per_frame = []

        if self.cap is None:
            self._close_video()
            return {
                "total_persons_detected": 0,
                "total_objects_detected": 0,
                "person_ratio": 0.0,
                "object_ratio": 0.0,
                "avg_persons_per_frame": 0.0,
                "avg_objects_per_frame": 0.0,
            }

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % (self.sample_rate * 3) != 0:
                continue

            try:
                results = self.yolo_model(frame, verbose=False)

                frame_persons = 0
                frame_objects = 0

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        if conf > 0.5:
                            if cls == person_id:
                                person_count += 1
                                frame_persons += 1
                            else:
                                object_count += 1
                                frame_objects += 1

                detections_per_frame.append(
                    {"persons": frame_persons, "objects": frame_objects}
                )
            except Exception as e:
                print(f"YOLO error on frame {frame_count}: {e}")

        self._close_video()

        total_detections = person_count + object_count
        person_ratio = person_count / total_detections if total_detections > 0 else 0.0
        object_ratio = object_count / total_detections if total_detections > 0 else 0.0

        avg_persons_per_frame = (
            person_count / len(detections_per_frame) if detections_per_frame else 0.0
        )
        avg_objects_per_frame = (
            object_count / len(detections_per_frame) if detections_per_frame else 0.0
        )

        return {
            "total_persons_detected": person_count,
            "total_objects_detected": object_count,
            "person_ratio": round(float(person_ratio), 3),
            "object_ratio": round(float(object_ratio), 3),
            "avg_persons_per_frame": round(float(avg_persons_per_frame), 2),
            "avg_objects_per_frame": round(float(avg_objects_per_frame), 2),
        }

    def extract_all_features(
        self,
        include_cuts: bool = True,
        include_motion: bool = True,
        include_text: bool = True,
        include_objects: bool = True,
    ) -> Dict:

        self._open_video()
        fps = float(self.fps) if self.fps is not None else 0.0
        duration = float(self.duration) if self.duration is not None else 0.0
        total_frames = int(self.total_frames) if self.total_frames is not None else 0

        metadata = {
            "video_path": str(self.video_path),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "sample_rate": self.sample_rate,
        }
        self._close_video()

        features = {"metadata": metadata}

        if include_cuts:
            features["shot_cuts"] = self.detect_shot_cuts()

        if include_motion:
            features["motion_analysis"] = self.analyze_motion()

        if include_text:
            features["text_detection"] = self.detect_text()

        if include_objects and self.yolo_model:
            features["object_detection"] = self.detect_objects_and_people()

        return features

    def save_features(self, features: Dict, output_path: str):
        with open(output_path, "w") as f:
            json.dump(features, f, indent=2, cls=NumpyEncoder)
        print(f"Features saved to: {output_path}")


def analyze_videos(video_paths, yolo_model_version: str, output_dir: str = "."):

    results = {}

    for video_path in video_paths:
        try:
            video_name = Path(video_path).stem
            output_json = (
                f"{output_dir}/{video_name}_features({yolo_model_version}).json"
            )

            extractor = VideoFeatureExtractor(
                video_path, yolo_model_version=yolo_model_version, sample_rate=5
            )

            features = extractor.extract_all_features(
                include_cuts=True,
                include_motion=True,
                include_text=True,
                include_objects=True,
            )

            extractor.save_features(features, str(output_json))
            results[video_name] = features

        except FileNotFoundError:
            results[video_name] = {"error": f"File not found: {video_path}"}
        except Exception as e:
            results[video_name] = {"error": str(e)}

    return results


if __name__ == "__main__":

    video_folder = Path("./videos")
    video_files = [str(f) for f in video_folder.glob("*") if f.is_file()]
    yolo_model_version = "yolo12l"  # change between n, s, m, l, x as needed

    if video_files:
        results = analyze_videos(video_files, yolo_model_version)
    else:
        print(f"No files found in {video_folder}")
