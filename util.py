import warnings
import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import Any, List
from pathlib import Path
import torch
import sys
from huggingface_hub import hf_hub_download

# 过滤不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 模型目录
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

class FaceRecognitionONNX:
    def __init__(self, hf_token=None):
        self.detection_model = None
        self.recognition_session = None
        self.initialized = False
        self.hf_token = hf_token
        self.retinaface_model = None
        
    def download(self, repo_id, path, HF_TOKEN=None):
        """优化下载逻辑"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # 检查必要文件
            essential_files = ['config.json', 'model.safetensors']
            existing_files = [f for f in essential_files if os.path.exists(os.path.join(path, f))]
            
            if len(existing_files) >= 2:
                logger.info("Essential model files exist, skipping download")
                return True
            
            # 只下载必要文件
            files_to_download = ['config.json', 'model.safetensors', 'wrapper.py']
            
            for file in files_to_download:
                full_path = os.path.join(path, file)
                if not os.path.exists(full_path):
                    try:
                        hf_hub_download(repo_id, file, token=HF_TOKEN, 
                                      local_dir=path)
                        logger.debug(f"Downloaded {file}")
                    except Exception as e:
                        if "404" in str(e):
                            logger.debug(f"File {file} not available")
                        else:
                            logger.warning(f"Download failed for {file}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def load_model_from_local_path(self, path, HF_TOKEN=None):
        """从本地路径加载模型"""
        try:
            if not os.path.exists(path):
                return None
            
            # 检查必要文件
            required_files = ['config.json']
            if not all(os.path.exists(os.path.join(path, f)) for f in required_files):
                return None
            
            # 保存当前状态
            cwd = os.getcwd()
            original_path = sys.path.copy()
            
            try:
                os.chdir(path)
                sys.path.insert(0, path)
                
                from transformers import AutoModel
                
                model = AutoModel.from_pretrained(
                    os.path.abspath(path), 
                    trust_remote_code=True, 
                    token=HF_TOKEN,
                    local_files_only=True
                )
                return model
                
            except Exception as e:
                logger.debug(f"Model loading attempt failed: {e}")
                return None
            finally:
                os.chdir(cwd)
                sys.path = original_path
                
        except Exception as e:
            logger.error(f"Load from local failed: {e}")
            return None

    def initialize_models(self):
        """优化模型初始化"""
        try:
            logger.info("Initializing face recognition models...")
            
            # 1. 尝试加载RetinaFace
            retinaface_loaded = self._load_retinaface()
            
            # 2. 加载ONNX识别模型
            recognition_path = MODELS_DIR / "webface_r50.onnx"
            if recognition_path.exists():
                try:
                    self.recognition_session = ort.InferenceSession(str(recognition_path))
                    logger.info("ONNX recognition model loaded")
                except Exception as e:
                    logger.warning(f"ONNX load failed: {e}")
            
            self.initialized = True
            self.check_model_health()
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._initialize_fallback()
    
    def _load_retinaface(self):
        """加载RetinaFace模型"""
        cache_path = MODELS_DIR / "retinaface_cache"
        
        # 检查缓存
        if cache_path.exists():
            model_files = list(cache_path.glob("*"))
            if len(model_files) > 3:  # 有足够文件
                logger.info("Loading RetinaFace from cache")
                self.retinaface_model = self.load_model_from_local_path(cache_path, self.hf_token)
                if self.retinaface_model:
                    self.detection_model = self.retinaface_model
                    return True
        
        # 下载并加载
        logger.info("Downloading RetinaFace model")
        repo_id = "minchul/private_retinaface_resnet50"
        cache_path.mkdir(exist_ok=True)
        
        if self.download(repo_id, cache_path, self.hf_token):
            self.retinaface_model = self.load_model_from_local_path(cache_path, self.hf_token)
            if self.retinaface_model:
                self.detection_model = self.retinaface_model
                return True
        
        logger.warning("RetinaFace loading failed, using fallback")
        return False
    
    def check_model_health(self):
        """模型健康检查"""
        status = {
            'retinaface': self.retinaface_model is not None,
            'recognition': self.recognition_session is not None
        }
        logger.info(f"Model status: {status}")
        return status
    
    def _initialize_fallback(self):
        """后备方案"""
        logger.info("Using fallback mode")
        self.detection_model = None
        self.initialized = True

# 全局实例
face_onnx = FaceRecognitionONNX()

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp the provided face image using the supplied homography matrix.
    Typically used to align faces prior to embedding extraction.
    """
    try:
        if isinstance(image, bytes):
            image = bytes_to_image(image)
        
        if homography_matrix is None:
            logger.warning("Homography matrix is None, returning original image")
            return image
        
        # 确保homography_matrix是numpy数组
        if not isinstance(homography_matrix, np.ndarray):
            homography_matrix = np.array(homography_matrix, dtype=np.float32)
        
        # 检查矩阵形状
        if homography_matrix.shape != (3, 3):
            logger.warning(f"Invalid homography matrix shape: {homography_matrix.shape}, expected (3, 3)")
            return image
        
        # 获取图像尺寸
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        
        # 使用透视变换进行人脸对齐
        warped_face = cv2.warpPerspective(image, homography_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        return warped_face
        
    except Exception as e:
        logger.error(f"Face warping failed: {e}")
        return image

def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces within the provided image.
    """
    if not face_onnx.initialized:
        face_onnx.initialize_models()
    
    if isinstance(image, bytes):
        image = bytes_to_image(image)
    
    if face_onnx.retinaface_model is not None:
        return _detect_faces_retinaface(image)
    else:
        return _detect_faces_opencv(image)

def _detect_faces_retinaface(image: np.ndarray) -> List[Any]:
    """使用RetinaFace模型检测人脸 - 修复版本"""
    try:
        if face_onnx.retinaface_model is None:
            raise ValueError("RetinaFace model not available")
        
        # 图像预处理
        from torchvision.transforms import Compose, ToTensor, Normalize
        from PIL import Image
        
        # 转换BGR到RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 预处理变换
        trans = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        input_tensor = trans(pil_img).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            outputs = face_onnx.retinaface_model(input_tensor)
            
        # 详细调试输出
        logger.info(f"Model output type: {type(outputs)}")
        logger.info(f"Number of outputs: {len(outputs)}")
        
        # 检查每个输出的详细信息
        for i, output in enumerate(outputs):
            if output is None:
                logger.info(f"Output {i}: None")
            elif hasattr(output, 'shape'):
                logger.info(f"Output {i}: shape={output.shape}, type={type(output)}")
            elif hasattr(output, '__len__'):
                logger.info(f"Output {i}: length={len(output)}, type={type(output)}")
            else:
                logger.info(f"Output {i}: value={output}, type={type(output)}")
        
        # 安全解包
        if len(outputs) >= 6:
            aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = outputs
            
            # 检查关键输出的具体内容
            logger.info(f"Score details: {score}")
            logger.info(f"Bbox details: {bbox}")
            logger.info(f"Landmarks details: {orig_ldmks}")
            
            # 检查score的具体值和形状
            if score is not None:
                if hasattr(score, 'shape'):
                    logger.info(f"Score shape: {score.shape}")
                if hasattr(score, '__len__'):
                    logger.info(f"Score length: {len(score)}")
                    if len(score) > 0:
                        logger.info(f"First score value: {score[0]}")
                        logger.info(f"Score values: {score}")
        
        # 处理检测结果
        faces = []
        h, w = image.shape[:2]
        
        # 更灵活的检测结果处理
        if score is not None and hasattr(score, '__len__') and len(score) > 0:
            # 检查是否有超过阈值的检测
            valid_detections = 0
            
            for i in range(len(score)):
                try:
                    # 安全获取score值
                    current_score = score[i] if hasattr(score, '__getitem__') else score
                    if hasattr(current_score, 'item'):
                        score_value = current_score.item()
                    else:
                        score_value = float(current_score)
                    
                    logger.info(f"Processing detection {i} with score: {score_value}")
                    
                    # 调整阈值，因为模型可能使用不同的评分标准
                    threshold = 0.5  # 可以尝试调整为0.1或0.01进行测试
                    
                    if score_value > threshold:
                        face_obj = type('Face', (), {})()
                        
                        # 尝试多种方式获取bbox
                        bbox_np = None
                        
                        # 方法1: 直接使用bbox输出
                        if bbox is not None and hasattr(bbox, '__len__') and len(bbox) > i:
                            bbox_value = bbox[i]
                            if hasattr(bbox_value, 'detach'):
                                bbox_np = bbox_value.detach().numpy()
                            else:
                                bbox_np = np.array(bbox_value)
                            logger.info(f"Using direct bbox: {bbox_np}")
                        
                        # 方法2: 如果bbox无效，从关键点计算
                        if bbox_np is None and orig_ldmks is not None and hasattr(orig_ldmks, '__len__') and len(orig_ldmks) > i:
                            landmarks_value = orig_ldmks[i]
                            if hasattr(landmarks_value, 'detach'):
                                landmarks_np = landmarks_value.detach().numpy()
                            else:
                                landmarks_np = np.array(landmarks_value)
                            
                            if landmarks_np.size >= 10:  # 5个关键点
                                landmarks_np = landmarks_np.reshape(-1, 2)
                                # 从关键点计算边界框
                                x_min, y_min = landmarks_np.min(axis=0)
                                x_max, y_max = landmarks_np.max(axis=0)
                                # 扩大边界框
                                width = x_max - x_min
                                height = y_max - y_min
                                x_min = max(0, x_min - width * 0.1)
                                y_min = max(0, y_min - height * 0.1)
                                x_max = min(1, x_max + width * 0.1)
                                y_max = min(1, y_max + height * 0.1)
                                bbox_np = np.array([x_min, y_min, x_max, y_max])
                                logger.info(f"Calculated bbox from landmarks: {bbox_np}")
                        
                        # 方法3: 使用默认边界框（整个图像）
                        if bbox_np is None:
                            bbox_np = np.array([0.1, 0.1, 0.9, 0.9])  # 使用图像中心区域
                            logger.info(f"Using default bbox: {bbox_np}")
                        
                        # 转换bbox坐标到像素值
                        if bbox_np.size >= 4:
                            if bbox_np.size == 4:
                                x1, y1, x2, y2 = bbox_np
                            else:
                                x1, y1, x2, y2 = bbox_np[:4]
                            
                            # 检查bbox坐标是否在合理范围内
                            if x1 < 0 or y1 < 0 or x2 > 2 or y2 > 2:
                                logger.warning(f"Unusual bbox coordinates: {x1}, {y1}, {x2}, {y2}")
                                # 假设坐标是归一化的，但可能范围是[0,1]或[-0.5,1.5]等
                                # 尝试标准化到[0,1]范围
                                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)
                            
                            # 转换为像素坐标
                            x1_px, y1_px = int(x1 * w), int(y1 * h)
                            x2_px, y2_px = int(x2 * w), int(y2 * h)
                            
                            # 确保坐标在图像范围内
                            x1_px, y1_px = max(0, x1_px), max(0, y1_px)
                            x2_px, y2_px = min(w, x2_px), min(h, y2_px)
                            
                            face_obj.bbox = np.array([x1_px, y1_px, x2_px, y2_px])
                            logger.info(f"Final bbox in pixels: {face_obj.bbox}")
                        else:
                            logger.warning(f"Invalid bbox format: {bbox_np}")
                            continue
                        
                        # 处理关键点
                        if orig_ldmks is not None and hasattr(orig_ldmks, '__len__') and len(orig_ldmks) > i:
                            landmarks_value = orig_ldmks[i]
                            if hasattr(landmarks_value, 'detach'):
                                landmarks_np = landmarks_value.detach().numpy()
                            else:
                                landmarks_np = np.array(landmarks_value)
                            
                            if landmarks_np.size >= 10:  # 5个关键点
                                landmarks_np = landmarks_np.reshape(-1, 2)
                                # 转换为像素坐标
                                landmarks_np = landmarks_np * np.array([w, h])
                                face_obj.landmarks = landmarks_np
                                logger.info(f"Using model landmarks: {face_obj.landmarks}")
                            else:
                                # 使用默认关键点
                                face_obj.landmarks = get_default_landmarks(face_obj.bbox)
                                logger.info(f"Using default landmarks: {face_obj.landmarks}")
                        else:
                            # 使用默认关键点
                            face_obj.landmarks = get_default_landmarks(face_obj.bbox)
                            logger.info(f"Using default landmarks: {face_obj.landmarks}")
                        
                        face_obj.score = score_value
                        faces.append(face_obj)
                        valid_detections += 1
                        logger.info(f"Successfully added face detection {i}")
                    else:
                        logger.info(f"Detection {i} below threshold: {score_value} <= {threshold}")
                        
                except Exception as e:
                    logger.error(f"Error processing detection {i}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Found {valid_detections} valid face detections")
        
        if len(faces) == 0:
            logger.info("No faces detected by RetinaFace, using OpenCV fallback")
            return _detect_faces_opencv(image)
            
        return faces
        
    except Exception as e:
        logger.error(f"RetinaFace face detection failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return _detect_faces_opencv(image)

def get_default_landmarks(bbox):
    """根据边界框生成默认的关键点"""
    x1, y1, x2, y2 = bbox
    w_face = x2 - x1
    h_face = y2 - y1
    return np.array([
        [x1 + w_face * 0.3, y1 + h_face * 0.3],  # 左眼
        [x1 + w_face * 0.7, y1 + h_face * 0.3],  # 右眼
        [x1 + w_face * 0.5, y1 + h_face * 0.5],  # 鼻子
        [x1 + w_face * 0.3, y1 + h_face * 0.7],  # 左嘴角
        [x1 + w_face * 0.7, y1 + h_face * 0.7]   # 右嘴角
    ])

def _detect_faces_opencv(image: np.ndarray) -> List[Any]:
    """使用OpenCV后备方案检测人脸"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        rects = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        faces = []
        for (x, y, w, h) in rects:
            face_obj = type('Face', (), {})()
            face_obj.bbox = np.array([x, y, x+w, y+h])
            face_obj.landmarks = np.array([
                [x + w*0.3, y + h*0.3],
                [x + w*0.7, y + h*0.3],
                [x + w*0.5, y + h*0.5],
                [x + w*0.3, y + h*0.7],
                [x + w*0.7, y + h*0.7]
            ])
            face_obj.score = 0.8
            faces.append(face_obj)
        
        return faces
    except Exception as e:
        logger.error(f"OpenCV face detection failed: {e}")
        return []

def compute_face_embedding(face_image: Any) -> np.ndarray:
    """Compute a numerical embedding vector for the provided face image."""
    if not face_onnx.initialized:
        face_onnx.initialize_models()
    
    if isinstance(face_image, bytes):
        face_image = bytes_to_image(face_image)
    
    if isinstance(face_image, np.ndarray):
        if face_onnx.recognition_session is not None:
            return _compute_embedding_mobilefacenet(face_image)
        else:
            return _compute_embedding_simple(face_image)
    
    raise ValueError("Invalid face image input")

def _compute_embedding_mobilefacenet(face_image: np.ndarray) -> np.ndarray:
    """使用MobileFaceNet计算人脸嵌入"""
    try:
        # 先进行人脸对齐
        aligned_face = align_face_landmarks(face_image)
        
        # 然后进行特征提取
        input_data = preprocess_face_mobilefacenet(aligned_face)
        input_name = face_onnx.recognition_session.get_inputs()[0].name
        embedding = face_onnx.recognition_session.run(None, {input_name: input_data})[0]
        return embedding.flatten()
    except Exception as e:
        logger.error(f"MobileFaceNet embedding failed: {e}")
        return _compute_embedding_simple(face_image)

def align_face_landmarks(face_image: np.ndarray) -> np.ndarray:
    """使用关键点进行人脸对齐"""
    try:
        # 检测人脸关键点
        landmarks = detect_face_keypoints(face_image)
        
        # 标准人脸关键点位置（用于对齐）
        dst_points = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # 计算仿射变换矩阵
        transform = cv2.estimateAffinePartial2D(landmarks, dst_points, method=cv2.RANSAC)[0]
        
        if transform is None:
            logger.warning("Affine transformation failed, using resize")
            return cv2.resize(face_image, (112, 112))
        
        # 将仿射变换矩阵转换为单应性矩阵（3x3）
        homography_matrix = np.vstack([transform, [0, 0, 1]])
        
        # 使用warp_face函数进行人脸对齐
        aligned_face = warp_face(face_image, homography_matrix)
        
        # 裁剪到标准大小
        aligned_face = cv2.resize(aligned_face, (112, 112))
        
        return aligned_face
        
    except Exception as e:
        logger.error(f"Face alignment failed: {e}")
        return cv2.resize(face_image, (112, 112))

def preprocess_face_mobilefacenet(face_image: np.ndarray) -> np.ndarray:
    """MobileFaceNet专用预处理"""
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    face_normalized = (face_normalized - mean) / std
    input_data = np.transpose(face_normalized, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def _compute_embedding_simple(face_image: np.ndarray) -> np.ndarray:
    """简化的特征提取 (后备方案)"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture = np.concatenate([sobelx.ravel()[:32], sobely.ravel()[:32]])
    return np.concatenate([hist, texture])

def detect_face_keypoints(face_image: Any) -> Any:
    """Identify facial keypoints (landmarks) for alignment or analysis."""
    if isinstance(face_image, np.ndarray):
        faces = detect_faces(face_image)
        if faces:
            return faces[0].landmarks
    
    h, w = face_image.shape[:2] if isinstance(face_image, np.ndarray) else (112, 112)
    return np.array([
        [w * 0.3, h * 0.3], [w * 0.7, h * 0.3], [w * 0.5, h * 0.5],
        [w * 0.3, h * 0.7], [w * 0.7, h * 0.7]
    ])

def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """End-to-end pipeline that returns a similarity score between two faces."""
    try:
        if isinstance(image_a, bytes):
            img_a = bytes_to_image(image_a)
        else:
            img_a = image_a
            
        if isinstance(image_b, bytes):
            img_b = bytes_to_image(image_b)
        else:
            img_b = image_b
        
        faces_a = detect_faces(img_a)
        faces_b = detect_faces(img_b)
        
        if not faces_a or not faces_b:
            logger.warning("No faces detected in one or both images")
            return 0.0
        
        face_a = faces_a[0]
        face_b = faces_b[0]
        
        # 提取人脸区域
        bbox_a = face_a.bbox.astype(int)
        x1, y1, x2, y2 = bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]
        h, w = img_a.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_region_a = img_a[y1:y2, x1:x2]
        
        bbox_b = face_b.bbox.astype(int)
        x1, y1, x2, y2 = bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3]
        h, w = img_b.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_region_b = img_b[y1:y2, x1:x2]
        
        if face_region_a.size == 0 or face_region_b.size == 0:
            logger.warning("Invalid face regions")
            return 0.0
        
        # 计算人脸嵌入向量
        embedding_a = compute_face_embedding(face_region_a)
        embedding_b = compute_face_embedding(face_region_b)
        
        # 计算余弦相似度
        embedding_a_norm = embedding_a / (np.linalg.norm(embedding_a) + 1e-10)
        embedding_b_norm = embedding_b / (np.linalg.norm(embedding_b) + 1e-10)
        similarity = np.dot(embedding_a_norm, embedding_b_norm)
        
        logger.info(f"Similarity: {similarity:.3f}")
        return float(max(0.0, min(1.0, similarity)))
        
    except Exception as e:
        logger.error(f"Face similarity calculation failed: {e}")
        return 0.0