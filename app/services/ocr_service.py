"""OCR (Optical Character Recognition) service for extracting text from images."""
import io
import re
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import numpy as np

# Try to import EasyOCR - better for social media content
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCRService:
    """Service for performing OCR on images."""
    
    @staticmethod
    def ocr_image_bytes(b: bytes) -> str:
        """
        Extract text from image bytes using OCR with image preprocessing.
        
        Args:
            b: Image bytes
            
        Returns:
            Extracted text string
        """
        img = Image.open(io.BytesIO(b))
        
        # Convert to RGB first if needed (handle RGBA, P mode, etc.)
        if img.mode != 'RGB':
            # Create a white background for transparent images
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            else:
                img = img.convert('RGB')
        
        # Convert to grayscale
        img = img.convert("L")
        
        # Upscale image for better OCR (especially for small text)
        # Scale to at least 300 DPI equivalent (2x-3x scaling helps with small text)
        width, height = img.size
        if width < 2000 or height < 2000:
            # Upscale by factor to ensure minimum dimensions
            scale_factor = max(2000 / width, 2000 / height, 2.5)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Enhance contrast to improve binarization (try multiple levels)
        enhancer = ImageEnhance.Contrast(img)
        # Try multiple contrast levels - higher contrast often helps Instagram images
        img = enhancer.enhance(2.5)  # Increase contrast
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Apply sharpening filter
        img = img.filter(ImageFilter.SHARPEN)
        
        # Try adaptive thresholding if numpy/scipy available, otherwise use fixed threshold
        img_array = np.array(img)
        
        # Use adaptive thresholding - works better for varying lighting
        try:
            from scipy import ndimage
            # Calculate local mean for adaptive thresholding
            kernel_size = 35
            local_mean = ndimage.uniform_filter(img_array.astype(np.float32), size=kernel_size)
            # Adaptive threshold: pixel is white if > local_mean - 10
            img_array = np.where(img_array > (local_mean - 10), 255, 0).astype(np.uint8)
        except ImportError:
            # Fallback to Otsu-like threshold or fixed threshold
            # Calculate dynamic threshold based on image statistics
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
            threshold = max(128, min(180, mean_brightness - std_brightness * 0.5))
            img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        
        # Convert back to PIL Image
        img = Image.fromarray(img_array)
        
        # Try EasyOCR first if available (often better for Instagram/social media)
        if EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR reader (use GPU if available, otherwise CPU)
                reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                
                # Convert PIL Image to numpy array for EasyOCR
                img_array = np.array(img)
                
                # Run OCR
                results = reader.readtext(img_array, paragraph=True)
                
                # Combine all detected text
                easyocr_text = "\n".join([result[1] for result in results if result[2] > 0.3])  # Confidence > 0.3
                
                # If EasyOCR produced reasonable results (more than 10 characters), use it
                if len(easyocr_text.strip()) > 10:
                    best_text = easyocr_text
                    # Post-process
                    best_text = OCRService._post_process_text(best_text)
                    return best_text
            except Exception:
                # Fall back to Tesseract if EasyOCR fails
                pass
        
        # Try multiple Tesseract OCR configs and pick the best result
        # Different PSM modes work better for different image layouts
        configs = [
            r'--oem 3 --psm 11',  # Sparse text (one word per line) - often best for Instagram
            r'--oem 3 --psm 6',   # Uniform block of text
            r'--oem 3 --psm 12',  # OSD sparse text (with orientation detection)
            r'--oem 3 --psm 4',   # Single column of text
            r'--oem 3 --psm 3',   # Fully automatic (default)
        ]
        
        best_text = ""
        max_words = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config)
                # Count words as a simple quality metric
                word_count = len(text.split())
                if word_count > max_words:
                    max_words = word_count
                    best_text = text
            except Exception:
                continue
        
        # Fallback to default if all configs fail
        if not best_text:
            try:
                best_text = pytesseract.image_to_string(img)
            except Exception:
                return ""
        
        # Post-process: fix common OCR misreadings for exercise labels
        best_text = OCRService._post_process_text(best_text)
        return best_text
    
    @staticmethod
    def _post_process_text(text: str) -> str:
        """
        Post-process OCR text to fix common misreadings.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Post-processed text
        """
        # Fix "82:" -> "B2:" (B is often misread as 8)
        text = re.sub(r'\b82([:\-])', r'B2\1', text)
        # Fix similar misreadings for other exercise numbers
        text = re.sub(r'\b81([:\-])', r'B1\1', text)
        text = re.sub(r'\b83([:\-])', r'B3\1', text)
        text = re.sub(r'\b72([:\-])', r'A2\1', text)
        text = re.sub(r'\b71([:\-])', r'A1\1', text)
        text = re.sub(r'\b73([:\-])', r'A3\1', text)
        # Context-aware correction: if we see B1 followed by 82, correct to B2
        text = re.sub(r'(\bB1[:\-].*?\n.*?)82([:\-])', r'\1B2\2', text, flags=re.MULTILINE | re.IGNORECASE)
        # Same for other letter patterns (A1->82=A2, C1->82=C2, etc.)
        text = re.sub(r'(\bA1[:\-].*?\n.*?)72([:\-])', r'\1A2\2', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'(\bC1[:\-].*?\n.*?)82([:\-])', r'\1C2\2', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'(\bD1[:\-].*?\n.*?)82([:\-])', r'\1D2\2', text, flags=re.MULTILINE | re.IGNORECASE)
        # Ensure spaces are preserved around colons and X multipliers
        # Add space after colon if missing: "A1:GOOD" -> "A1: GOOD"
        text = re.sub(r'([A-E]\d*):([A-Z])', r'\1: \2', text)
        # Add space before X when followed by number: "GOODX10" -> "GOOD X10"
        text = re.sub(r'([A-Za-z])X(\d)', r'\1 X\2', text)
        return text
    
    @staticmethod
    def ocr_many_images_to_text(dir_with_pngs: str) -> str:
        """
        Extract text from multiple PNG images in a directory.
        
        Args:
            dir_with_pngs: Directory path containing PNG images
            
        Returns:
            Combined text from all images
        """
        import glob
        import os
        texts = []
        for img_path in sorted(glob.glob(os.path.join(dir_with_pngs, "frame_*.png"))):
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                    txt = OCRService.ocr_image_bytes(image_bytes)
                    if txt.strip():
                        texts.append(txt)
            except Exception:
                continue
        return "\n".join(texts)

