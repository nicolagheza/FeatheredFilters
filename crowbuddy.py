import cv2
import time
import logging
import os
import signal
import sys
import argparse
import RPi.GPIO as GPIO
from edgetpumodel import EdgeTPUModel
from utils import plot_one_box, Colors

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
NATURE_PIN = 21
TRASH_PIN = 26
SCORE_THRESHOLD = 0.85
BLOCK_DURATION = 3  # seconds
SAVE_IMAGES_DIR = "detections"  # Directory to save detection images


class CrowBuddy:
    def __init__(
        self,
        model_path,
        names_file="models/crowbuddy.yaml",
        test_mode=False,
        save_detections=True,
    ):
        self.in_blocking_state = False
        self.block_start_time = 0
        self.test_mode = test_mode
        self.save_detections = save_detections
        self.colors = Colors()  # For visualization

        # if save_detections
        if self.save_detections:
            os.makedirs(SAVE_IMAGES_DIR, exist_ok=True)
            logger.info(f"Detection images will be saved to {SAVE_IMAGES_DIR}/")

        # Initialize GPIO only if not in test mode
        if not self.test_mode:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(NATURE_PIN, GPIO.OUT)
            GPIO.setup(TRASH_PIN, GPIO.OUT)
        else:
            logger.info("Running in test mode - GPIO operations disabled")

        # Load EdgeTPU model using your existing EdgeTPUModel class
        try:
            self.model = EdgeTPUModel(
                model_path,
                names_file,
                conf_thresh=SCORE_THRESHOLD,
                iou_thresh=0.3,
                filter_classes=None,
                agnostic_nms=False,
                max_det=100,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Failed to open camera")
        logger.info("Camera initialized successfully")

    def check_blocking_state(self):
        """Check if system is in blocking state and update if needed"""
        if self.in_blocking_state:
            if time.time() - self.block_start_time >= BLOCK_DURATION:
                self.in_blocking_state = False
                logger.debug("Blocking state released")
        return self.in_blocking_state

    def trigger_gpio(self, class_id):
        """Trigger appropriate GPIO based on detection class"""
        try:
            pin = NATURE_PIN if class_id == 0 else TRASH_PIN

            if not self.test_mode:
                GPIO.output(pin, True)
                time.sleep(0.1)
                GPIO.output(pin, False)
            else:
                logger.info(f"TEST MODE: Would trigger GPIO pin {pin}")

            self.in_blocking_state = True
            self.block_start_time = time.time()
            logger.info(
                f"{'TEST MODE: ' if self.test_mode else ''}Triggered GPIO pin {pin} for class {int(class_id)}"
            )
        except Exception as e:
            logger.error(f"Error triggering GPIO: {e}")

    def save_detection_image(self, frame, detections, timestamp):
        """Save frame with detection boxes draw on it"""
        # Make a copy to avoid modifying the original frame
        output_image = frame.copy()

        for detection in detections:
            xyxy = detection[:4].astype(int)
            conf = float(detection[4])
            cls = int(detection[5])

            # Draw bonding box
            label = f"{self.model.names[cls]} {conf:.2f}"
            output_image = plot_one_box(
                xyxy, output_image, label=label, color=self.colors(cls, True)
            )

        # Create filename with timestamp
        filename = os.path.join(SAVE_IMAGES_DIR, f"detection_{timestamp}.jpg")

        # Save the image
        cv2.imwrite(filename, output_image)
        logger.debug(f"Saved detection image to {filename}")
        return filename

    def main_loop(self):
        """Main detection loop"""
        try:
            logger.info("Starting detection loop")
            # Get input size from model
            input_size = self.model.get_image_size()

            while True:
                if self.check_blocking_state():
                    time.sleep(0.1)  # Small delay during blocking
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Process frame using get_image_tensor from utils
                from utils import get_image_tensor

                full_image, net_image, pad = get_image_tensor(frame, input_size[0])

                # Use model's forward method directly
                pred = self.model.forward(net_image)

                # Process predictions to get detections
                detections = pred[0]  # First batch

                # Find best detection (highest confidence)
                best_detection = None
                best_confidence = 0

                for detection in detections:
                    # Assuming detection format is [x1, y1, x2, y2, conf, cls]
                    conf = float(detection[4])
                    if conf > best_confidence and conf > SCORE_THRESHOLD:
                        best_detection = detection
                        best_confidence = conf

                if best_detection is not None:
                    cls = int(best_detection[5])
                    logger.info(
                        f"Detected: {'nature' if cls == 0 else 'trash'} "
                        f"(confidence: {best_confidence:.2f})"
                    )

                    # Save image with detections if enabled
                    if self.save_detections:
                        # Scale detection coordinates to match the original frame
                        scaled_detections = detections.copy()
                        scaled_detections[:, :4] = self.model.get_scaled_coords(
                            scaled_detections[:, :4], frame, pad
                        )
                        timestamp = int(time.time())
                        saved_path = self.save_detection_image(
                            frame, scaled_detections, timestamp
                        )
                        logger.info(f"Saved detection image to {saved_path}")

                    self.trigger_gpio(cls)

                time.sleep(0.01)  # Small delay to prevent CPU overload

        except KeyboardInterrupt:
            logger.info("Detection loop stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in detection loop: {e}")
            logger.exception("Full traceback:")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        self.cap.release()
        if not self.test_mode:
            GPIO.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="CrowBuddy Detection System")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode without GPIO"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/crow-7.5M-int8_edgetpu.tflite",
        help="Path to EdgeTPU model file",
    )
    parser.add_argument(
        "--names", type=str, default="models/crowbuddy.yaml", help="Path to names file"
    )
    parser.add_argument(
        "--save-detections",
        action="store_true",
        default=True,
        help="Save images when detections occur",
    )
    parser.add_argument(
        "--no-save-detections",
        action="store_false",
        dest="save_detections",
        help="Disable saving detection images",
    )
    return parser.parse_args()


def signal_handler(sig, frame):
    """Handle SIGINT and SIGTERM gracefully"""
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        buddy = CrowBuddy(
            args.model,
            args.names,
            test_mode=args.test,
            save_detections=args.save_detections,
        )
        buddy.main_loop()
    except Exception as e:
        logger.error(f"CrowBuddy startup failed: {e}")
        sys.exit(1)
