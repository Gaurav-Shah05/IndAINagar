import cv2
import torch
import numpy as np
import time
import os
import sys

# zerodce directory -> Python path
zerodce_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dependencies', 'enhancement', 'zerodce'))
if zerodce_path not in sys.path:
    sys.path.append(zerodce_path)


class VideoEnhancer:
    def __init__(self, model_path, scale_factor=1, device=None):
        """
        Initialize the video enhancer with the specified model and device.
        
        Args:
            model_path (str): Path to the enhancement model weights
            scale_factor (int): Scale factor for image processing
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
        """
        # Determine device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Convert string device to torch.device
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
            
        print(f"Using device: {self.device} for enhancement")
        
        # Set scale factor
        self.scale_factor = scale_factor
        
        # Import model from zerodce directory
        try:
            # Import components from zerodce
            from modeling import model
            from utils import scale_image, get_device
            
            # Save the scale_image function for later use
            self.scale_image = scale_image
            
            # Initialize the enhancement model
            self.net = model.enhance_net_nopool(scale_factor).to(self.device)
            
            # Load model weights
            if os.path.exists(model_path):
                self.net.load_state_dict(torch.load(model_path, map_location=self.device))
                self.net.eval()
                self.model_loaded = True
                print(f"Enhancement model loaded from {model_path}")
            else:
                self.model_loaded = False
                print(f"Warning: Enhancement model not found at {model_path}")
        except ImportError as e:
            print(f"Warning: Failed to import from zerodce: {e}")
            print(f"Python path: {sys.path}")
            self.model_loaded = False

    def calculate_brightness(self, frame):
        """Calculate the average brightness of a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def enhance_frame(self, frame):
        """Enhance a single frame using the loaded model or a fallback method"""
        if not self.model_loaded:
            # Fallback enhancement (simple brightness/contrast adjustment)
            return self._fallback_enhance(frame)
        
        # Convert frame from BGR to RGB and create a tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float().div(255)
        
        # Handle scaling
        if hasattr(self, 'scale_image'):
            # Use the imported scale_image function
            frame_tensor = self.scale_image(frame_tensor, self.scale_factor, self.device)
        else:
            # Fallback scaling implementation
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Run inference on the frame
        with torch.no_grad():
            enhanced_tensor, _ = self.net(frame_tensor)
        
        # Process the enhanced tensor
        enhanced_tensor = enhanced_tensor.squeeze(0).cpu().detach()
        enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
        enhanced_img = enhanced_tensor.mul(255).byte().permute(1, 2, 0).numpy()
        
        # Convert enhanced image from RGB to BGR (for OpenCV)
        enhanced_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        
        # Apply denoising
        processed_frame = cv2.GaussianBlur(enhanced_bgr, (5, 5), 0)
        
        return processed_frame
    
    def _fallback_enhance(self, frame):
        """Simple enhancement method when the model is not available"""
        # Create a copy of the frame
        enhanced = frame.copy()
        
        # Increase brightness and contrast
        alpha = 1.5  # Contrast control (1.0 means no change)
        beta = 30    # Brightness control (0 means no change)
        
        # Apply brightness/contrast adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # Reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return enhanced
    
    def process_frame(self, frame, mode="Auto", threshold=30):
        """
        Process a frame according to the enhancement mode
        
        Args:
            frame: Input video frame
            mode (str): Enhancement mode - "Off", "On", or "Auto"
            threshold (int): Brightness threshold for auto mode
            
        Returns:
            Processed frame
        """
        if mode == "Off":
            return frame
        
        elif mode == "On":
            return self.enhance_frame(frame)
        
        elif mode == "Auto":
            brightness = self.calculate_brightness(frame)
            if brightness < threshold:
                return self.enhance_frame(frame)
            else:
                return frame
        
        return frame  # Default fallback