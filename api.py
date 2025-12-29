"""
SAM 2 & SAM 3 Image Segmentation API + Sam-3d-objects 3D Generation

This API provides:
1. Image segmentation using Meta's Segment Anything Model 2 (SAM 2)
2. Image segmentation using Meta's Segment Anything Model 3 (SAM 3)
3. 3D object generation from masks using Sam-3d-objects

Endpoints:
- /segment: Get segmentation mask from a single point (SAM 2)
- /segment-binary: Get segmentation mask with mask context support (SAM 2)
- /segment-sam3d: Get segmentation mask from a single point (SAM 3)
- /segment-binary-sam3d: Get segmentation mask with mask context support (SAM 3)
- /generate-3d: Generate 3D Gaussian splat from image and mask
"""

import os

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing torch/spconv
# These must be set BEFORE any imports that use spconv
# ============================================================================
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

# Set spconv environment variables early (before any imports)
os.environ["SPCONV_TUNE_DEVICE"] = "0"
os.environ["SPCONV_ALGO_TIME_LIMIT"] = "100"  # Set to 100ms (was 0 = infinite tuning)
os.environ["TORCH_CUDA_ARCH_LIST"] = "all"

# Prevent thread explosion - limit OpenMP threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import io
import base64
import numpy as np
import torch

# ============================================================================
# CRITICAL: Set PyTorch default dtype to float32 IMMEDIATELY
# This MUST be done before any other imports to prevent spconv float64 errors
# ============================================================================
torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import cv2
import json
import tempfile
import sys
import subprocess
import uuid
from typing import List, Dict, Optional
from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# SAM 2 imports from transformers
from transformers import Sam2Processor, Sam2Model

# ============================================================================
# PYTORCH CONFIGURATION FOR SPCONV COMPATIBILITY
# Set default float dtype to float32 to prevent algorithm tuning errors
# ============================================================================
torch.set_default_dtype(torch.float32)

# Sam-3d-objects imports (optional - gracefully fail if not available)
try:
    sam3d_notebook_path = "./sam-3d-objects/notebook"
    if os.path.exists(sam3d_notebook_path):
        sys.path.insert(0, sam3d_notebook_path)
        from inference import Inference

        print(f"✓ Sam-3d-objects imported successfully")
    else:
        print(f"⚠ Sam-3d-objects notebook path not found at {sam3d_notebook_path}")
except Exception as e:
    print(f"⚠ Sam-3d-objects import failed: {e}")

# Configure device
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Initialize FastAPI app
app = FastAPI(
    title="SAM 2 & SAM 3 Image Segmentation API",
    description="Segment objects in images using Segment Anything Model 2 (SAM 2) and SAM 3, plus 3D generation with SAM-3D-Objects",
    version="2.0.0",
)

# Create assets folder for downloadable files
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Mount assets folder as static files (accessible at /assets/)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Global model and processor instances (SAM 2)
model = None
processor = None

# Global model and processor instances (SAM 3)
sam3_model = None
sam3_processor = None
SAM3_AVAILABLE = False

# Task storage for async 3D generation
generation_tasks: Dict[str, Dict] = {}


def initialize_model():
    """Initialize SAM 2 model and processor from Hugging Face"""
    global model, processor

    try:
        model_id = "facebook/sam2.1-hiera-large"
        print(f"Loading SAM 2 model from {model_id}...")

        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(device)

        print("✓ SAM 2 model and processor initialized successfully")

    except Exception as e:
        print(f"✗ Error initializing SAM 2 model: {e}")
        raise


def initialize_sam3_model():
    """Initialize SAM 3 model and processor from facebookresearch/sam3"""
    global sam3_model, sam3_processor, SAM3_AVAILABLE

    try:
        # Try to import SAM3 from the cloned repository
        sam3_path = "./sam3"
        if os.path.exists(sam3_path):
            sys.path.insert(0, sam3_path)
            print(f"SAM3 path found: {sam3_path}")
        else:
            print(f"⚠ SAM3 path not found at {sam3_path}")

        sam3_loaded = False
        build_sam3_image_model = None

        # Try to find build_sam3_image_model in various locations
        import_paths = [
            ("sam3.sam3", "build_sam3_image_model"),
            ("sam3", "build_sam3_image_model"),
            ("sam3.build_sam3", "build_sam3_image_model"),
            ("sam3.model_builder", "build_sam3_image_model"),
        ]

        for module_path, func_name in import_paths:
            try:
                module = __import__(module_path, fromlist=[func_name])
                if hasattr(module, func_name):
                    build_sam3_image_model = getattr(module, func_name)
                    print(f"✓ Found {func_name} in {module_path}")
                    break
            except ImportError:
                continue

        # Method 1: Use build_sam3_image_model if found
        if build_sam3_image_model is not None and not sam3_loaded:
            try:
                print("Loading SAM 3 via build_sam3_image_model...")
                
                # Get function signature to understand parameters
                import inspect
                sig = inspect.signature(build_sam3_image_model)
                print(f"  Function signature: {sig}")

                # Check for local checkpoint
                checkpoint_paths = [
                    "./sam3/checkpoints/sam3.pt",
                    "./sam3/checkpoints/sam3_hiera_large.pt",
                    "./sam3/checkpoints/sam3_hiera_base_plus.pt",
                ]
                checkpoint_path = None
                for cp in checkpoint_paths:
                    if os.path.exists(cp):
                        checkpoint_path = cp
                        print(f"  Found checkpoint: {checkpoint_path}")
                        break

                # Use correct parameter names based on function signature
                device_str = "cuda" if device.type == "cuda" else "cpu"
                
                # Build kwargs based on available parameters
                kwargs = {}
                param_names = list(sig.parameters.keys())
                
                if 'device' in param_names:
                    kwargs['device'] = device_str
                if 'checkpoint_path' in param_names and checkpoint_path:
                    kwargs['checkpoint_path'] = checkpoint_path
                if 'load_from_HF' in param_names:
                    kwargs['load_from_HF'] = checkpoint_path is None
                if 'enable_segmentation' in param_names:
                    kwargs['enable_segmentation'] = True
                if 'eval_mode' in param_names:
                    kwargs['eval_mode'] = True
                
                print(f"  Calling with kwargs: {kwargs}")
                sam3_model = build_sam3_image_model(**kwargs)

                # Model is already on device and in eval mode from build function
                # Try to find a predictor wrapper (Sam3Processor from examples)
                predictor_found = False
                predictor_paths = [
                    "sam3.model.sam3_image_processor.Sam3Processor",  # From sam3_agent.ipynb example
                    "sam3.sam3_image_processor.Sam3Processor",
                    "sam3.sam3_image_predictor",
                    "sam3.predictor",
                    "sam3.SAM3ImagePredictor",
                    "sam3.SAM3Predictor",
                ]
                
                for pred_path in predictor_paths:
                    try:
                        if "." in pred_path:
                            module_path, class_name = pred_path.rsplit(".", 1)
                            module = __import__(module_path, fromlist=[class_name])
                            if hasattr(module, class_name):
                                PredictorClass = getattr(module, class_name)
                                # Sam3Processor takes (model, confidence_threshold) according to example
                                if "Sam3Processor" in class_name:
                                    sam3_processor = PredictorClass(sam3_model, confidence_threshold=0.5)
                                else:
                                    sam3_processor = PredictorClass(sam3_model)
                                predictor_found = True
                                print(f"  ✓ Found predictor: {pred_path}")
                                # Check if it has set_image method (simpler API)
                                if hasattr(sam3_processor, 'set_image'):
                                    print(f"    Processor has 'set_image' method - will use simpler API")
                                break
                    except (ImportError, AttributeError, TypeError) as e:
                        print(f"    Failed to load {pred_path}: {e}")
                        continue
                
                if not predictor_found:
                    # Use model directly - we'll handle this in the routes
                    sam3_processor = sam3_model
                    print("  Using model directly (no separate predictor)")
                
                sam3_loaded = True
                print("✓ SAM 3 model loaded successfully via build_sam3_image_model")
                
                # Print available methods for debugging
                model_methods = [m for m in dir(sam3_model) if not m.startswith('_') and callable(getattr(sam3_model, m, None))]
                print(f"  Available model methods: {model_methods[:20]}...")
                
                # Check forward signature
                if hasattr(sam3_model, 'forward'):
                    import inspect
                    try:
                        forward_sig = inspect.signature(sam3_model.forward)
                        print(f"  Forward signature: {forward_sig}")
                    except:
                        pass
                
            except Exception as e1:
                print(f"⚠ SAM 3 build_sam3_image_model failed: {e1}")
                import traceback
                traceback.print_exc()

        # Method 2: Try to find any build function in sam3 module
        if not sam3_loaded:
            try:
                import sam3
                print(f"SAM3 module contents: {dir(sam3)}")
                
                # Look for build functions
                build_funcs = [attr for attr in dir(sam3) if 'build' in attr.lower()]
                print(f"  Build functions found: {build_funcs}")
                
                for func_name in build_funcs:
                    try:
                        func = getattr(sam3, func_name)
                        if callable(func):
                            print(f"  Trying {func_name}...")
                            sam3_model = func()
                            sam3_model = sam3_model.to(device)
                            sam3_model.eval()
                            sam3_processor = sam3_model
                            sam3_loaded = True
                            print(f"✓ SAM 3 model initialized via sam3.{func_name}")
                            break
                    except Exception as e:
                        print(f"    {func_name} failed: {e}")
                        continue
                        
            except ImportError as e2:
                print(f"⚠ SAM 3 module import failed: {e2}")
            except Exception as e2:
                print(f"⚠ SAM 3 module exploration failed: {e2}")

        # Method 3: Try HuggingFace (requires access approval)
        if not sam3_loaded:
            try:
                from transformers import AutoProcessor, AutoModel
                sam3_model_id = "facebook/sam3"
                print(f"Loading SAM 3 from HuggingFace: {sam3_model_id}...")
                print("  Note: This model is GATED. You need to:")
                print("    1. Request access at https://huggingface.co/facebook/sam3")
                print("    2. Run: huggingface-cli login")
                sam3_processor = AutoProcessor.from_pretrained(sam3_model_id)
                sam3_model = AutoModel.from_pretrained(sam3_model_id).to(device)
                sam3_loaded = True
                print("✓ SAM 3 loaded from HuggingFace")
            except Exception as e3:
                print(f"⚠ SAM 3 HuggingFace load failed: {e3}")

        # Set availability based on actual loading success
        SAM3_AVAILABLE = sam3_loaded

        if not sam3_loaded:
            print("=" * 60)
            print("✗ SAM 3 could not be loaded. SAM3D routes will return 503 errors.")
            print("")
            print("SAM3 is installed but failed to load. Possible issues:")
            print("  1. Missing checkpoint file in sam3/checkpoints/")
            print("  2. Missing dependencies")
            print("")
            print("Try downloading the checkpoint:")
            print("  huggingface-cli download facebook/sam3 --local-dir ./sam3/checkpoints/")
            print("=" * 60)
            sam3_model = None
            sam3_processor = None

    except Exception as e:
        print(f"✗ SAM 3 initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print("✗ SAM3D routes will not be available")
        SAM3_AVAILABLE = False
        sam3_model = None
        sam3_processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on API startup"""
    initialize_model()
    initialize_sam3_model()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    sam2_ready = model is not None and processor is not None
    sam3_ready = SAM3_AVAILABLE and sam3_model is not None and sam3_processor is not None

    return {
        "status": "healthy" if sam2_ready else "degraded",
        "sam2": {
            "loaded": sam2_ready,
            "model_id": "facebook/sam2.1-hiera-large",
            "routes": ["/segment", "/segment-binary"],
        },
        "sam3": {
            "loaded": sam3_ready,
            "model_id": "facebook/sam3" if sam3_ready else None,
            "routes": ["/segment-sam3d", "/segment-binary-sam3d"],
            "error": None if sam3_ready else "SAM3 not installed. Run: git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .",
        },
        "device": str(device),
    }


class SegmentRequest(BaseModel):
    image: str  # base64 encoded image
    x: float  # X coordinate
    y: float  # Y coordinate
    multimask_output: bool = True  # Whether to return multiple masks
    mask_threshold: float = (
        0.0  # Threshold for mask logits (default: 0.0, use 0.5 for stricter)
    )
    invert_mask: bool = (
        False  # Whether to invert the mask (0=foreground, 255=background)
    )


@app.post("/segment")
async def segment_image(request: SegmentRequest):
    """
    Segment an object in an image based on a point coordinate.

    Args:
        request: JSON body containing:
            - image: Base64 encoded image string
            - x: X coordinate of the point (horizontal position)
            - y: Y coordinate of the point (vertical position)
            - multimask_output: Whether to return multiple mask predictions (default: True)

    Returns:
        JSON response containing:
        - masks: The segmentation masks as arrays
        - scores: Quality scores for each mask
        - input_point: The input point coordinate
        - image_shape: Dimensions of the input image
    """
    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=500, content={"error": "Model not initialized"}
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image_pil)

        # Prepare input points and labels in the format expected by the processor
        # Format: [[[[x, y]]]] - 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
        input_points = [[[[request.x, request.y]]]]
        input_labels = [[[1]]]  # 1 for positive click, 0 for negative click

        # Process inputs
        inputs = processor(
            images=image_pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process masks
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        # Convert masks to list and get scores
        mask_list = []
        scores = (
            outputs.iou_preds[0].cpu().numpy().tolist()
            if hasattr(outputs, "iou_preds")
            else [0.95] * masks.shape[0]
        )

        for i in range(masks.shape[0]):
            mask = masks[i].numpy()
            # Squeeze extra dimensions and ensure 2D
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                mask = mask[0] if mask.ndim > 2 else mask

            # Threshold mask
            mask = (mask > request.mask_threshold).astype(np.uint8) * 255

            # Apply morphological smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = (mask > 127).astype(np.uint8) * 255

            # Invert if requested
            if request.invert_mask:
                mask = 255 - mask

            mask_image = Image.fromarray(mask, mode="L")
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            buffer.seek(0)
            mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mask_list.append(
                {
                    "mask": mask_base64,
                    "mask_shape": mask.shape,
                    "score": float(scores[i]) if i < len(scores) else 0.95,
                }
            )

        return JSONResponse(
            {
                "success": True,
                "masks": mask_list,
                "input_point": [request.x, request.y],
                "image_shape": [image_pil.height, image_pil.width],
            }
        )

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ============================================================================
# SAM 3 SEGMENTATION ROUTES
# ============================================================================


class SegmentSam3dRequest(BaseModel):
    image: str  # base64 encoded image
    x: float  # X coordinate
    y: float  # Y coordinate
    multimask_output: bool = True  # Whether to return multiple masks
    mask_threshold: float = (
        0.0  # Threshold for mask logits (default: 0.0, use 0.5 for stricter)
    )
    invert_mask: bool = (
        False  # Whether to invert the mask (0=foreground, 255=background)
    )


@app.post("/segment-sam3d")
async def segment_image_sam3d(request: SegmentSam3dRequest):
    """
    Segment an object in an image based on a point coordinate using SAM 3.

    Args:
        request: JSON body containing:
            - image: Base64 encoded image string
            - x: X coordinate of the point (horizontal position)
            - y: Y coordinate of the point (vertical position)
            - multimask_output: Whether to return multiple mask predictions (default: True)
            - mask_threshold: Threshold for mask logits (default: 0.0)
            - invert_mask: Whether to invert the mask (default: False)

    Returns:
        JSON response containing:
        - masks: The segmentation masks as base64 PNG images with scores
        - input_point: The input point coordinate
        - image_shape: Dimensions of the input image
        - model: The model used for segmentation
    """
    try:
        if not SAM3_AVAILABLE or sam3_model is None or sam3_processor is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "SAM 3 model not available. Please install sam3.",
                    "instructions": [
                        "Option A - Local: git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .",
                        "Option B - HuggingFace: Request access at https://huggingface.co/facebook/sam3, then huggingface-cli login",
                    ],
                },
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image_pil)

        # Prepare point coordinates
        point_coords = np.array([[request.x, request.y]])
        point_labels = np.array([1])  # 1 for positive click

        # Determine which API to use based on what's available
        mask_list = []

        if hasattr(sam3_processor, 'set_image'):
            # Sam3Processor API - uses set_image() which returns inference_state
            # Then use set_text_prompt() or other methods for prompts
            try:
                # Check available methods
                processor_methods = [m for m in dir(sam3_processor) if not m.startswith('_') and callable(getattr(sam3_processor, m))]
                print(f"Sam3Processor methods: {processor_methods}")
                
                # set_image returns inference_state
                inference_state = sam3_processor.set_image(image_pil)
                print(f"✓ Image set, inference_state type: {type(inference_state)}")
                
                # Try to find method for point prompts
                # Sam3Processor has add_geometric_prompt for points
                if hasattr(sam3_processor, 'add_geometric_prompt'):
                    # add_geometric_prompt likely takes point coordinates and labels
                    # Try different parameter formats
                    import inspect
                    try:
                        sig = inspect.signature(sam3_processor.add_geometric_prompt)
                        print(f"add_geometric_prompt signature: {sig}")
                    except:
                        pass
                    
                    try:
                        # Try with state as first positional argument
                        output = sam3_processor.add_geometric_prompt(
                            inference_state,
                            point_coords=point_coords,  # [[x, y]]
                            point_labels=point_labels,   # [1] for positive
                        )
                        print(f"✓ Used add_geometric_prompt (format 1), output type: {type(output)}")
                    except Exception as e:
                        print(f"⚠ add_geometric_prompt format 1 failed: {e}, trying format 2...")
                        try:
                            # Try with state as keyword argument
                            output = sam3_processor.add_geometric_prompt(
                                state=inference_state,
                                point_coords=point_coords,
                                point_labels=point_labels,
                            )
                            print(f"✓ Used add_geometric_prompt (format 2), output type: {type(output)}")
                        except Exception as e2:
                            print(f"⚠ add_geometric_prompt format 2 failed: {e2}, trying format 3...")
                            try:
                                # Try with points/labels instead of point_coords/point_labels
                                output = sam3_processor.add_geometric_prompt(
                                    inference_state,
                                    points=point_coords,
                                    labels=point_labels,
                                )
                                print(f"✓ Used add_geometric_prompt (format 3), output type: {type(output)}")
                            except Exception as e3:
                                print(f"⚠ All add_geometric_prompt formats failed. Last error: {e3}")
                                raise e3
                elif hasattr(sam3_processor, 'set_point_prompt'):
                    output = sam3_processor.set_point_prompt(
                        state=inference_state,
                        point_coords=point_coords,
                        point_labels=point_labels,
                    )
                elif hasattr(sam3_processor, 'set_prompt'):
                    output = sam3_processor.set_prompt(
                        state=inference_state,
                        point_coords=point_coords,
                        point_labels=point_labels,
                    )
                elif hasattr(sam3_processor, 'predict'):
                    # Fallback to predict if available
                    with torch.no_grad():
                        masks, scores, logits = sam3_processor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=request.multimask_output,
                        )
                    output = {"masks": masks, "scores": scores}
                else:
                    # No point prompt methods available
                    raise AttributeError("Sam3Processor doesn't have point prompt methods. Available methods: " + str(processor_methods))
                
                # Extract masks from output
                if isinstance(output, dict):
                    masks = output.get("masks", None)
                    scores = output.get("scores", output.get("iou_scores", None))
                elif hasattr(output, 'masks'):
                    masks = output.masks
                    scores = getattr(output, 'scores', None)
                else:
                    masks = output
                    scores = None
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"✗ Sam3Processor API error: {error_trace}")
                print(f"  Falling back to direct model API (BatchedDatapoint)")
                # Fall back to direct model API - will be handled in elif block below
                masks = None
                scores = None
                
            if masks is not None:
                # Process masks from Sam3Processor
                # masks shape: [num_masks, H, W]
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                masks = np.squeeze(masks)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]
                    
                if scores is None:
                    scores = [0.95] * len(masks)
                elif isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy().flatten().tolist()
                elif not isinstance(scores, (list, tuple)):
                    scores = [0.95] * len(masks)
                    
                # Limit to requested number of masks
                if not request.multimask_output and len(masks) > 1:
                    best_idx = np.argmax(scores) if len(scores) > 0 else 0
                    masks = masks[best_idx:best_idx+1]
                    scores = [scores[best_idx]] if len(scores) > best_idx else [0.95]
                
                for i in range(len(masks)):
                    mask = masks[i]
                    mask = (mask > request.mask_threshold).astype(np.uint8) * 255

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    mask = (mask > 127).astype(np.uint8) * 255

                    if request.invert_mask:
                        mask = 255 - mask

                    mask_image = Image.fromarray(mask, mode="L")
                    buffer = io.BytesIO()
                    mask_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    mask_list.append({
                        "mask": mask_base64,
                        "mask_shape": list(mask.shape),
                        "score": float(scores[i]) if i < len(scores) else 0.95,
                    })
            else:
                # Sam3Processor failed, fall through to direct model API
                print(f"Sam3Processor didn't return masks, trying direct model API...")
        
        # Use direct model API if Sam3Processor failed or doesn't have set_image
        if not mask_list and (hasattr(sam3_model, 'forward') and (sam3_processor is sam3_model or not hasattr(sam3_processor, 'set_image'))):
            # Direct model API (no separate predictor)
            # SAM3 uses BatchedDatapoint interface with specific structure
            try:
                from sam3.model.data_misc import BatchedDatapoint, FindStage, BatchedFindTarget, BatchedInferenceMetadata
                from torchvision.transforms import functional as TF
                
                # Prepare image tensor - SAM3 expects img_batch
                image_tensor = TF.to_tensor(image_pil).unsqueeze(0).to(device)
                
                # Prepare point prompts - normalize coordinates to [0, 1]
                h, w = image_np.shape[:2]
                # input_points: [batch, num_points, 2] - normalized coordinates
                input_points = torch.tensor([[[request.x / w, request.y / h]]], dtype=torch.float32, device=device)  # [1, 1, 2]
                # input_points_mask: [batch, num_points] - 1s for valid points
                input_points_mask = torch.ones((1, 1), dtype=torch.bool, device=device)  # [1, 1]
                
                # Create FindStage with all required fields
                # FindStage requires: img_ids, text_ids, input_boxes, input_boxes_mask, 
                # input_boxes_label, input_points, input_points_mask
                img_ids = torch.tensor([0], dtype=torch.long, device=device)  # Single image, ID 0
                text_ids = torch.empty((0,), dtype=torch.long, device=device)  # No text prompts
                input_boxes = torch.empty((1, 0, 4), dtype=torch.float32, device=device)  # No boxes
                input_boxes_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)  # No boxes
                input_boxes_label = torch.empty((1, 0), dtype=torch.long, device=device)  # No box labels
                
                find_stage = FindStage(
                    img_ids=img_ids,
                    text_ids=text_ids,
                    input_boxes=input_boxes,
                    input_boxes_mask=input_boxes_mask,
                    input_boxes_label=input_boxes_label,
                    input_points=input_points,
                    input_points_mask=input_points_mask,
                )
                print(f"✓ FindStage created successfully")
                
                # Create BatchedFindTarget (likely empty for segmentation)
                try:
                    find_target = BatchedFindTarget()
                except Exception as e:
                    print(f"⚠ BatchedFindTarget creation failed: {e}, using None")
                    find_target = None
                
                # Create BatchedInferenceMetadata
                try:
                    metadata = BatchedInferenceMetadata()
                except Exception as e:
                    print(f"⚠ BatchedInferenceMetadata creation failed: {e}, using None")
                    metadata = None
                
                # Create BatchedDatapoint with correct structure
                batched_datapoint = BatchedDatapoint(
                    img_batch=image_tensor,
                    find_text_batch=[],  # Empty list for no text prompts
                    find_inputs=[find_stage],
                    find_targets=[find_target] if find_target else [],
                    find_metadatas=[metadata] if metadata else [],
                    raw_images=[image_pil],  # Original PIL image
                )
                print(f"✓ BatchedDatapoint created successfully")
                
                with torch.no_grad():
                    print(f"Calling sam3_model.forward with BatchedDatapoint...")
                    outputs = sam3_model(batched_datapoint)
                    print(f"✓ Model forward call successful, output type: {type(outputs)}")
                    if hasattr(outputs, '__dict__'):
                        print(f"  Output attributes: {list(outputs.__dict__.keys())}")
                    elif isinstance(outputs, dict):
                        print(f"  Output keys: {list(outputs.keys())}")
                    elif isinstance(outputs, (list, tuple)):
                        print(f"  Output length: {len(outputs)}")
                        for i, item in enumerate(outputs):
                            print(f"    Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A') if hasattr(item, 'shape') else 'N/A'}")
                    
            except ImportError as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"✗ Import error: {error_trace}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"SAM3 import failed: {str(e)}",
                        "message": "SAM3 model structure may have changed. Please check SAM3 documentation.",
                        "trace": error_trace
                    },
                )
            except Exception as e:
                # Fallback: try to understand the error and provide helpful message
                import traceback
                error_trace = traceback.format_exc()
                print(f"✗ SAM3 forward error: {error_trace}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"SAM3 model forward call failed: {str(e)}",
                        "trace": error_trace,
                        "message": "SAM3 model requires BatchedDatapoint with specific structure. Please check SAM3 documentation for correct usage."
                    },
                )

            # Extract masks from output
            # SAM3 output structure may vary - try to extract masks
            masks = None
            scores = None
            
            print(f"Attempting to extract masks from output type: {type(outputs)}")
            
            if isinstance(outputs, dict):
                print(f"  Output is dict, keys: {list(outputs.keys())}")
                masks = outputs.get("masks", outputs.get("pred_masks", outputs.get("mask", None)))
                scores = outputs.get("iou_predictions", outputs.get("scores", outputs.get("iou_scores", None)))
            elif hasattr(outputs, 'masks'):
                print(f"  Output has 'masks' attribute")
                masks = outputs.masks
                scores = getattr(outputs, 'scores', None) or getattr(outputs, 'iou_scores', None)
            elif isinstance(outputs, (list, tuple)):
                print(f"  Output is list/tuple, length: {len(outputs)}")
                masks = outputs[0] if len(outputs) > 0 else None
                scores = outputs[1] if len(outputs) > 1 else None
            elif isinstance(outputs, torch.Tensor):
                print(f"  Output is Tensor, shape: {outputs.shape}")
                masks = outputs
            else:
                # Try to access as attribute
                print(f"  Trying to access as attributes")
                masks = getattr(outputs, 'masks', None) or getattr(outputs, 'pred_masks', None)
                scores = getattr(outputs, 'scores', None) or getattr(outputs, 'iou_scores', None)

            if masks is not None:
                print(f"✓ Masks extracted, type: {type(masks)}, shape: {getattr(masks, 'shape', 'N/A') if hasattr(masks, 'shape') else 'N/A'}")
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                masks = np.squeeze(masks)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]

                if scores is None:
                    scores = [0.95] * len(masks)
                elif isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy().flatten().tolist()
                elif not isinstance(scores, (list, tuple)):
                    scores = [0.95] * len(masks)

                # Limit to requested number of masks
                if not request.multimask_output and len(masks) > 1:
                    # Use the mask with highest score
                    best_idx = np.argmax(scores) if len(scores) > 0 else 0
                    masks = masks[best_idx:best_idx+1]
                    scores = [scores[best_idx]] if len(scores) > best_idx else [0.95]

                for i in range(len(masks)):
                    mask = masks[i]
                    mask = (mask > request.mask_threshold).astype(np.uint8) * 255

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    mask = (mask > 127).astype(np.uint8) * 255

                    if request.invert_mask:
                        mask = 255 - mask

                    mask_image = Image.fromarray(mask, mode="L")
                    buffer = io.BytesIO()
                    mask_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    mask_list.append({
                        "mask": mask_base64,
                        "mask_shape": list(mask.shape),
                        "score": float(scores[i]) if i < len(scores) else 0.95,
                    })
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "SAM3 model returned no masks. Model API may be incompatible.",
                        "output_type": str(type(outputs)),
                        "output_attrs": dir(outputs) if hasattr(outputs, '__dict__') else "N/A"
                    },
                )

        else:
            # HuggingFace Transformers API (only if processor is callable)
            if callable(sam3_processor):
                input_points = [[[[request.x, request.y]]]]
                input_labels = [[[1]]]

                inputs = sam3_processor(
                    images=image_pil,
                    input_points=input_points,
                    input_labels=input_labels,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = sam3_model(**inputs)

                masks = sam3_processor.post_process_masks(
                    outputs.pred_masks.cpu(), inputs["original_sizes"]
                )[0]
            else:
                # If processor is not callable (e.g., Sam3Processor), return error
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "SAM3 processor API not fully implemented for this processor type.",
                        "processor_type": str(type(sam3_processor)),
                        "message": "Sam3Processor doesn't support HuggingFace-style API. Please use point prompts via add_geometric_prompt or direct model API."
                    },
                )

            scores = (
                outputs.iou_preds[0].cpu().numpy().tolist()
                if hasattr(outputs, "iou_preds")
                else [0.95] * masks.shape[0]
            )

            for i in range(masks.shape[0]):
                mask = masks[i].numpy()
                mask = np.squeeze(mask)
                if mask.ndim != 2:
                    mask = mask[0] if mask.ndim > 2 else mask

                mask = (mask > request.mask_threshold).astype(np.uint8) * 255

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                mask = (mask > 127).astype(np.uint8) * 255

                if request.invert_mask:
                    mask = 255 - mask

                mask_image = Image.fromarray(mask, mode="L")
                buffer = io.BytesIO()
                mask_image.save(buffer, format="PNG")
                buffer.seek(0)
                mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                mask_list.append({
                    "mask": mask_base64,
                    "mask_shape": list(mask.shape),
                    "score": float(scores[i]) if i < len(scores) else 0.95,
                })

        return JSONResponse(
            {
                "success": True,
                "masks": mask_list,
                "input_point": [request.x, request.y],
                "image_shape": [image_pil.height, image_pil.width],
                "model": "sam3",
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})


class SegmentBinaryRequest(BaseModel):
    image: str  # base64 encoded image
    points: List[Dict[str, float]]  # [{"x": float, "y": float}, ...]
    previous_mask: Optional[str] = None  # base64 PNG of previous mask (optional)
    mask_threshold: float = 0.0  # Threshold for mask logits


@app.post("/segment-binary")
async def segment_image_binary(request: SegmentBinaryRequest):
    """
    Segment an image and return the mask as base64 encoded PNG.
    """
    try:
        if model is None or processor is None:
            return JSONResponse(
                status_code=500, content={"error": "Model not initialized"}
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Validate points
        if not request.points or len(request.points) == 0:
            return JSONResponse(
                status_code=400, content={"error": "At least one point required"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_pil_array = np.array(
            image_pil
        )  # Keep original image for color preservation

        # Convert points to the format expected by SAM 2
        # Process each point SEPARATELY to avoid losing segments when adding new points
        # Format: [[[[x, y]]]] - 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)

        # Collect masks from each point
        all_masks = []

        for point_idx, point in enumerate(request.points):

            # Process single point
            input_points = [[[[point["x"], point["y"]]]]]
            input_labels = [[[1]]]  # Positive point

            # Process inputs
            inputs = processor(
                images=image_pil,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(device)

            # Run inference for this single point
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process masks
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]

            # Get scores
            scores = (
                outputs.iou_preds[0].cpu().numpy()
                if hasattr(outputs, "iou_preds")
                else np.array([0.95] * masks.shape[0])
            )

            # Get best mask for this point
            best_mask_idx = np.argmax(scores)
            point_mask = masks[best_mask_idx].numpy()

            # Squeeze and ensure 2D
            point_mask = np.squeeze(point_mask)
            if point_mask.ndim != 2:
                point_mask = point_mask[0] if point_mask.ndim > 2 else point_mask

            # Apply threshold
            point_mask = (point_mask > request.mask_threshold).astype(np.uint8) * 255

            all_masks.append(point_mask)

        # Union all masks from all points
        mask = all_masks[0].copy()
        for i in range(1, len(all_masks)):
            mask = np.maximum(mask, all_masks[i])

        # Add previous mask to the union (accumulate)
        if request.previous_mask:
            try:
                mask_data = base64.b64decode(request.previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_array = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_array)
            except Exception:
                pass

        mask = (mask > request.mask_threshold).astype(np.uint8) * 255

        if request.previous_mask:
            try:
                mask_data = base64.b64decode(request.previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_np = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_np)
            except Exception:
                pass

        # Apply morphological smoothing (less aggressive to preserve thin regions from multiple points)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Only use CLOSE (fill small holes) - skip OPEN which can eliminate thin connections
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Light gaussian blur instead of heavy filtering
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = (mask > 127).astype(np.uint8) * 255

        # Check if mask is mostly white (inverted) - if mean > 127, invert it
        mask_mean = mask.mean()
        if mask_mean > 127:
            mask = 255 - mask

        # Verify dimensions match
        if image_pil_array.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image_pil_array.shape[1], image_pil_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert mask from 0-255 to 0-1 for multiplication
        mask_normalized = mask.astype(np.float32) / 255.0

        # Expand mask to 3 channels (R, G, B)
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Apply mask: foreground keeps original colors, background becomes black
        masked_image = (image_pil_array.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Convert to PNG and encode as base64
        masked_image_pil = Image.fromarray(masked_image, mode="RGB")
        buffer = io.BytesIO()
        masked_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        score = float(scores[best_mask_idx])

        return JSONResponse(
            {
                "success": True,
                "mask": mask_base64,
                "score": score,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})


class SegmentBinarySam3dRequest(BaseModel):
    image: str  # base64 encoded image
    points: List[Dict[str, float]]  # [{"x": float, "y": float}, ...]
    previous_mask: Optional[str] = None  # base64 PNG of previous mask (optional)
    mask_threshold: float = 0.0  # Threshold for mask logits


@app.post("/segment-binary-sam3d")
async def segment_image_binary_sam3d(request: SegmentBinarySam3dRequest):
    """
    Segment an image using SAM 3 and return the masked image as base64 encoded PNG.

    Args:
        request: JSON body containing:
            - image: Base64 encoded image string
            - points: List of point coordinates [{"x": float, "y": float}, ...]
            - previous_mask: Optional base64 PNG of previous mask to accumulate
            - mask_threshold: Threshold for mask logits (default: 0.0)

    Returns:
        JSON response containing:
        - mask: Base64 encoded PNG of the masked image (foreground preserved, background black)
        - score: Quality score of the segmentation
        - model: The model used for segmentation
    """
    try:
        if not SAM3_AVAILABLE or sam3_model is None or sam3_processor is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "SAM 3 model not available. Please install sam3.",
                    "instructions": [
                        "Option A - Local: git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .",
                        "Option B - HuggingFace: Request access at https://huggingface.co/facebook/sam3, then huggingface-cli login",
                    ],
                },
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid base64 image: {str(e)}"}
            )

        # Validate points
        if not request.points or len(request.points) == 0:
            return JSONResponse(
                status_code=400, content={"error": "At least one point required"}
            )

        # Process image
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_pil_array = np.array(image_pil)
        image_np = image_pil_array

        # Collect masks from each point
        all_masks = []
        best_score = 0.95

        # Determine which API to use based on what's available
        if hasattr(sam3_processor, 'set_image'):
            # SAM3ImagePredictor API (local installation with predictor)
            sam3_processor.set_image(image_np)

            for point in request.points:
                point_coords = np.array([[point["x"], point["y"]]])
                point_labels = np.array([1])  # Positive point

                with torch.no_grad():
                    masks, scores, logits = sam3_processor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                    )

                best_idx = np.argmax(scores)
                point_mask = masks[best_idx]
                best_score = float(scores[best_idx])

                point_mask = (point_mask > request.mask_threshold).astype(np.uint8) * 255
                all_masks.append(point_mask)

        elif hasattr(sam3_model, 'forward') and sam3_processor is sam3_model:
            # Direct model API (no separate predictor)
            # SAM3 uses BatchedDatapoint interface with specific structure
            try:
                from sam3.model.data_misc import BatchedDatapoint, FindStage, BatchedFindTarget, BatchedInferenceMetadata
                from torchvision.transforms import functional as TF

                # Prepare image tensor
                image_tensor = TF.to_tensor(image_pil).unsqueeze(0).to(device)
                h, w = image_np.shape[:2]

                for point in request.points:
                    # Normalize coordinates to [0, 1]
                    # input_points: [batch, num_points, 2] - normalized coordinates
                    input_points = torch.tensor([[[point["x"] / w, point["y"] / h]]], dtype=torch.float32, device=device)  # [1, 1, 2]
                    # input_points_mask: [batch, num_points] - 1s for valid points
                    input_points_mask = torch.ones((1, 1), dtype=torch.bool, device=device)  # [1, 1]

                    # Create FindStage with all required fields
                    img_ids = torch.tensor([0], dtype=torch.long, device=device)  # Single image, ID 0
                    text_ids = torch.empty((0,), dtype=torch.long, device=device)  # No text prompts
                    input_boxes = torch.empty((1, 0, 4), dtype=torch.float32, device=device)  # No boxes
                    input_boxes_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)  # No boxes
                    input_boxes_label = torch.empty((1, 0), dtype=torch.long, device=device)  # No box labels
                    
                    find_stage = FindStage(
                        img_ids=img_ids,
                        text_ids=text_ids,
                        input_boxes=input_boxes,
                        input_boxes_mask=input_boxes_mask,
                        input_boxes_label=input_boxes_label,
                        input_points=input_points,
                        input_points_mask=input_points_mask,
                    )
                    
                    # Create BatchedFindTarget and BatchedInferenceMetadata
                    try:
                        find_target = BatchedFindTarget()
                    except Exception as e:
                        print(f"⚠ BatchedFindTarget creation failed: {e}, using None")
                        find_target = None
                    
                    try:
                        metadata = BatchedInferenceMetadata()
                    except Exception as e:
                        print(f"⚠ BatchedInferenceMetadata creation failed: {e}, using None")
                        metadata = None

                    # Create BatchedDatapoint
                    batched_datapoint = BatchedDatapoint(
                        img_batch=image_tensor,
                        find_text_batch=[],
                        find_inputs=[find_stage],
                        find_targets=[find_target] if find_target else [],
                        find_metadatas=[metadata] if metadata else [],
                        raw_images=[image_pil],
                    )

                    with torch.no_grad():
                        print(f"Calling sam3_model.forward with BatchedDatapoint for point ({point['x']}, {point['y']})...")
                        outputs = sam3_model(batched_datapoint)
                        print(f"✓ Model forward call successful, output type: {type(outputs)}")
                        if hasattr(outputs, '__dict__'):
                            print(f"  Output attributes: {list(outputs.__dict__.keys())}")
                        elif isinstance(outputs, dict):
                            print(f"  Output keys: {list(outputs.keys())}")
                        elif isinstance(outputs, (list, tuple)):
                            print(f"  Output length: {len(outputs)}")

                    # Extract masks from output
                    masks = None
                    scores = None
                    
                    if isinstance(outputs, dict):
                        masks = outputs.get("masks", outputs.get("pred_masks", outputs.get("mask", None)))
                        scores = outputs.get("iou_predictions", outputs.get("scores", outputs.get("iou_scores", None)))
                    elif hasattr(outputs, 'masks'):
                        masks = outputs.masks
                        scores = getattr(outputs, 'scores', None) or getattr(outputs, 'iou_scores', None)
                    elif isinstance(outputs, (list, tuple)):
                        masks = outputs[0] if len(outputs) > 0 else None
                        scores = outputs[1] if len(outputs) > 1 else None
                    elif isinstance(outputs, torch.Tensor):
                        masks = outputs
                    else:
                        masks = getattr(outputs, 'masks', None) or getattr(outputs, 'pred_masks', None)
                        scores = getattr(outputs, 'scores', None) or getattr(outputs, 'iou_scores', None)

                    if masks is not None:
                        if isinstance(masks, torch.Tensor):
                            masks = masks.cpu().numpy()
                        masks = np.squeeze(masks)
                        if masks.ndim == 2:
                            masks = masks[np.newaxis, ...]

                        if scores is None:
                            scores = np.array([0.95])
                        elif isinstance(scores, torch.Tensor):
                            scores = scores.cpu().numpy().flatten()
                        elif not isinstance(scores, (list, tuple, np.ndarray)):
                            scores = np.array([0.95])

                        best_idx = np.argmax(scores) if len(scores) > 0 else 0
                        point_mask = masks[best_idx] if masks.ndim > 2 else masks
                        best_score = float(scores[best_idx]) if len(scores) > best_idx else 0.95

                        point_mask = (point_mask > request.mask_threshold).astype(np.uint8) * 255
                        all_masks.append(point_mask)
                    else:
                        print(f"⚠ Failed to extract mask from point {point}")
                        
            except ImportError as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"SAM3 BatchedDatapoint import failed: {str(e)}",
                        "message": "SAM3 model structure may have changed. Please check SAM3 documentation."
                    },
                )
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"⚠ Error processing points: {error_trace}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"SAM3 model forward call failed: {str(e)}",
                        "message": "SAM3 model requires BatchedDatapoint. Please check SAM3 documentation for correct usage."
                    },
                )

        else:
            # HuggingFace Transformers API (only if processor is callable)
            if callable(sam3_processor):
                for point in request.points:
                    input_points = [[[[point["x"], point["y"]]]]]
                    input_labels = [[[1]]]

                    inputs = sam3_processor(
                        images=image_pil,
                        input_points=input_points,
                        input_labels=input_labels,
                        return_tensors="pt",
                    ).to(device)

                    with torch.no_grad():
                        outputs = sam3_model(**inputs)

                    masks = sam3_processor.post_process_masks(
                        outputs.pred_masks.cpu(), inputs["original_sizes"]
                    )[0]
            else:
                # If processor is not callable, return error
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "SAM3 processor API not fully implemented for this processor type.",
                        "processor_type": str(type(sam3_processor)),
                        "message": "Sam3Processor doesn't support HuggingFace-style API for multiple points. Please use direct model API."
                    },
                )

                scores = (
                    outputs.iou_preds[0].cpu().numpy()
                    if hasattr(outputs, "iou_preds")
                    else np.array([0.95] * masks.shape[0])
                )

                best_idx = np.argmax(scores)
                point_mask = masks[best_idx].numpy()
                best_score = float(scores[best_idx])

                point_mask = np.squeeze(point_mask)
                if point_mask.ndim != 2:
                    point_mask = point_mask[0] if point_mask.ndim > 2 else point_mask

                point_mask = (point_mask > request.mask_threshold).astype(np.uint8) * 255
                all_masks.append(point_mask)

        # If no masks were collected, return an error
        if not all_masks:
            return JSONResponse(
                status_code=500,
                content={"error": "SAM3 model returned no masks. Model API may be incompatible."},
            )

        # Union all masks from all points
        mask = all_masks[0].copy()
        for i in range(1, len(all_masks)):
            mask = np.maximum(mask, all_masks[i])

        # Add previous mask to the union (accumulate)
        if request.previous_mask:
            try:
                mask_data = base64.b64decode(request.previous_mask)
                prev_mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                prev_mask_array = np.array(prev_mask_pil)
                mask = np.maximum(mask, prev_mask_array)
            except Exception:
                pass

        # Apply morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = (mask > 127).astype(np.uint8) * 255

        # Check if mask is mostly white (inverted) - if mean > 127, invert it
        mask_mean = mask.mean()
        if mask_mean > 127:
            mask = 255 - mask

        # Verify dimensions match
        if image_pil_array.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image_pil_array.shape[1], image_pil_array.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert mask from 0-255 to 0-1 for multiplication
        mask_normalized = mask.astype(np.float32) / 255.0

        # Expand mask to 3 channels (R, G, B)
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Apply mask: foreground keeps original colors, background becomes black
        masked_image = (image_pil_array.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Convert to PNG and encode as base64
        masked_image_pil = Image.fromarray(masked_image, mode="RGB")
        buffer = io.BytesIO()
        masked_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(
            {
                "success": True,
                "mask": mask_base64,
                "score": best_score,
                "model": "sam3",
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})


class Generate3dRequest(BaseModel):
    image: str  # base64 encoded image
    mask: str  # base64 encoded binary mask
    seed: int = 42


def _generate_3d_background(
    task_id: str, image_temp_path: str, mask_temp_path: str, seed: int
):
    """
    Background task for 3D generation.
    This function updates the generation_tasks dict with status and results.
    """
    ply_temp_path = None
    gif_temp_path = None

    try:
        generation_tasks[task_id]["status"] = "processing"

        # Create temp file for output PLY
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            ply_temp_path = tmp.name
            gif_temp_path = ply_temp_path.replace(".ply", ".gif")

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess_script = os.path.join(script_dir, "generate_3d_subprocess.py")

        print(f"[Task {task_id}] Running 3D generation in subprocess...")

        # Run subprocess
        result = subprocess.run(
            [
                sys.executable,
                subprocess_script,
                image_temp_path,
                mask_temp_path,
                str(seed),
                ply_temp_path,
                ASSETS_DIR,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Print subprocess output for debugging
        if result.stdout:
            print(f"[Task {task_id}][Subprocess stdout]:\n{result.stdout}")
        if result.stderr:
            print(f"[Task {task_id}][Subprocess stderr]:\n{result.stderr}")

        # Check if subprocess succeeded
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(
                f"[Task {task_id}] Subprocess failed with return code {result.returncode}"
            )

            generation_tasks[task_id]["status"] = "failed"
            generation_tasks[task_id]["error"] = error_msg
            return

        # Extract GIF data from subprocess output
        gif_b64 = None
        if "GIF_DATA_START" in result.stdout and "GIF_DATA_END" in result.stdout:
            try:
                start_idx = result.stdout.find("GIF_DATA_START") + len("GIF_DATA_START")
                end_idx = result.stdout.find("GIF_DATA_END")
                gif_b64 = result.stdout[start_idx:end_idx].strip()
                print(
                    f"[Task {task_id}] ✓ Extracted GIF: {len(gif_b64)} chars (base64)"
                )
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract GIF data: {e}")

        # Extract mesh URL from subprocess output
        mesh_url = None
        if "MESH_URL_START" in result.stdout and "MESH_URL_END" in result.stdout:
            try:
                start_idx = result.stdout.find("MESH_URL_START") + len("MESH_URL_START")
                end_idx = result.stdout.find("MESH_URL_END")
                mesh_url = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] ✓ Extracted mesh URL: {mesh_url}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract mesh URL: {e}")

        # Extract PLY URL from subprocess output
        ply_url = None
        if "PLY_URL_START" in result.stdout and "PLY_URL_END" in result.stdout:
            try:
                start_idx = result.stdout.find("PLY_URL_START") + len("PLY_URL_START")
                end_idx = result.stdout.find("PLY_URL_END")
                ply_url = result.stdout[start_idx:end_idx].strip()
                print(f"[Task {task_id}] ✓ Extracted PLY URL: {ply_url}")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not extract PLY URL: {e}")

        # Always read PLY as primary output
        ply_b64 = None
        ply_size_bytes = None

        if os.path.exists(ply_temp_path):
            print(f"[Task {task_id}] Reading PLY from {ply_temp_path}")
            with open(ply_temp_path, "rb") as f:
                ply_bytes = f.read()

            # Validate PLY header
            try:
                header_text = ply_bytes[: min(50000, len(ply_bytes))].decode(
                    "utf-8", errors="ignore"
                )
                if "end_header" not in header_text:
                    print(
                        f"[Task {task_id}] WARNING: PLY missing 'end_header' in first 50KB"
                    )
                    print(
                        f"[Task {task_id}] PLY appears to be binary, checking full file..."
                    )
                    # Check entire file
                    full_text = ply_bytes.decode("utf-8", errors="ignore")
                    if "end_header" not in full_text:
                        print(
                            f"[Task {task_id}] ERROR: PLY file corrupted or not ASCII format"
                        )
                    else:
                        print(
                            f"[Task {task_id}] Found end_header after 50KB - file is large but valid"
                        )
                else:
                    print(f"[Task {task_id}] ✓ PLY header valid (ASCII format)")
            except Exception as e:
                print(f"[Task {task_id}] Warning: Could not validate PLY header: {e}")

            ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
            ply_size_bytes = len(ply_bytes)
            print(f"[Task {task_id}] ✓ PLY loaded: {ply_size_bytes} bytes")

        # GIF data was already extracted from subprocess stdout above
        gif_size_bytes = len(gif_b64) if gif_b64 else None

        # Determine primary output (for backward compatibility)
        output_b64 = ply_b64 if ply_b64 else gif_b64
        output_type = "ply" if ply_b64 else "gif"
        output_size_bytes = ply_size_bytes if ply_b64 else gif_size_bytes

        if output_b64:
            print(
                f"[Task {task_id}] ✓ 3D generation successful ({output_type}): {output_size_bytes} bytes"
            )
        else:
            generation_tasks[task_id]["status"] = "failed"
            generation_tasks[task_id][
                "error"
            ] = "Neither GIF nor PLY file was generated"
            return

        generation_tasks[task_id]["status"] = "completed"
        generation_tasks[task_id]["output_b64"] = output_b64
        generation_tasks[task_id]["output_type"] = output_type
        generation_tasks[task_id]["output_size_bytes"] = output_size_bytes
        generation_tasks[task_id]["ply_b64"] = ply_b64
        generation_tasks[task_id]["ply_size_bytes"] = ply_size_bytes
        generation_tasks[task_id]["ply_url"] = ply_url
        generation_tasks[task_id]["gif_b64"] = gif_b64
        generation_tasks[task_id]["gif_size_bytes"] = gif_size_bytes
        generation_tasks[task_id]["mesh_url"] = mesh_url
        generation_tasks[task_id]["progress"] = 100

    except subprocess.TimeoutExpired:
        generation_tasks[task_id]["status"] = "failed"
        generation_tasks[task_id][
            "error"
        ] = "3D generation timed out (exceeded 10 minutes)"
    except Exception as e:
        print(f"[Task {task_id}] Error in 3D generation: {e}")
        import traceback

        traceback.print_exc()
        generation_tasks[task_id]["status"] = "failed"
        generation_tasks[task_id]["error"] = str(e)
    finally:
        # Clean up temporary files
        for path in [image_temp_path, mask_temp_path, ply_temp_path, gif_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    print(f"[Task {task_id}] Cleaned up temp file: {path}")
                except:
                    pass


@app.post("/generate-3d")
async def generate_3d(request: Generate3dRequest, background_tasks: BackgroundTasks):
    """
    Start 3D Gaussian splat generation (non-blocking, returns task ID).

    Returns immediately with a task_id that can be polled for results.
    This avoids gateway timeouts by returning immediately.

    Args:
        request: JSON body containing:
            - image: Base64 encoded RGB image (PNG or JPEG)
            - mask: Base64 encoded binary mask (0-1 grayscale)
            - seed: Random seed for reproducibility (default: 42)

    Returns:
        JSON response containing:
        - task_id: Unique ID to poll for results
        - status: "queued"
    """
    image_temp_path = None
    mask_temp_path = None

    try:
        # Decode base64 to temporary PNG files
        try:
            image_bytes = base64.b64decode(request.image)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_temp_path = tmp.name
                tmp.write(image_bytes)

            # Save for debugging
            image_pil_save = Image.open(image_temp_path).convert("RGB")
            image_pil_save.save("./test_img.png")
            print(f"✓ Saved incoming image as test_img.png")

            mask_bytes = base64.b64decode(request.mask)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                mask_temp_path = tmp.name
                tmp.write(mask_bytes)

            # Save for debugging
            mask_pil_save = Image.open(mask_temp_path).convert("L")
            mask_pil_save.save("./test_img_mask.png")
            print(f"✓ Saved incoming mask as test_img_mask.png")

        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid image or mask: {str(e)}"}
            )

        # Create unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task in storage
        generation_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "created_at": str(np.datetime64("now")),
        }

        # Add background task
        background_tasks.add_task(
            _generate_3d_background,
            task_id,
            image_temp_path,
            mask_temp_path,
            request.seed,
        )

        print(f"[API] Task {task_id} queued for 3D generation")

        return JSONResponse(
            {
                "success": True,
                "task_id": task_id,
                "status": "queued",
            }
        )

    except Exception as e:
        print(f"[API] Error creating 3D generation task: {e}")
        import traceback

        traceback.print_exc()

        # Clean up temp files on error
        for path in [image_temp_path, mask_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to queue 3D generation: {str(e)}"},
        )


@app.get("/generate-3d-status/{task_id}")
async def generate_3d_status(task_id: str):
    """
    Poll for 3D generation task status and results.

    Args:
        task_id: The task ID returned from /generate-3d

    Returns:
        JSON response containing:
        - status: "queued", "processing", "completed", or "failed"
        - progress: 0-100 (if applicable)
        - ply_b64: Base64 encoded PLY file (if completed)
        - error: Error message (if failed)
    """
    if task_id not in generation_tasks:
        return JSONResponse(
            status_code=404,
            content={"error": f"Task {task_id} not found"},
        )

    task = generation_tasks[task_id]

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
    }

    if task["status"] == "completed":
        response["ply_b64"] = task.get("output_b64")
        response["ply_size_bytes"] = task.get("output_size_bytes")
        response["gif_b64"] = task.get("gif_b64")
        response["gif_size_bytes"] = task.get("gif_size_bytes")
        response["mesh_url"] = task.get("mesh_url")

        # Encode mesh file to base64 if URL exists
        mesh_url = task.get("mesh_url")
        if mesh_url:
            mesh_filename = mesh_url.split("/")[-1]
            mesh_path = os.path.join(ASSETS_DIR, mesh_filename)

            # Detect mesh format from file extension
            if mesh_filename.endswith(".glb"):
                response["mesh_format"] = "glb"
            elif mesh_filename.endswith(".ply"):
                response["mesh_format"] = "ply"
            else:
                response["mesh_format"] = "unknown"

            if os.path.exists(mesh_path):
                try:
                    with open(mesh_path, "rb") as f:
                        mesh_bytes = f.read()
                    response["mesh_b64"] = base64.b64encode(mesh_bytes).decode("utf-8")
                    response["mesh_size_bytes"] = len(mesh_bytes)
                except Exception as e:
                    print(f"[API] Warning: Could not encode mesh to base64: {e}")
                    response["mesh_b64"] = None
                    response["mesh_size_bytes"] = 0
            else:
                print(f"[API] Warning: Mesh file not found at {mesh_path}")
                response["mesh_b64"] = None
                response["mesh_size_bytes"] = 0
        else:
            response["mesh_b64"] = None
            response["mesh_size_bytes"] = 0

        # Also include new naming convention
        response["output_b64"] = task.get("output_b64")
        response["output_type"] = task.get("output_type")  # "gif" or "ply"
        response["output_size_bytes"] = task.get("output_size_bytes")
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")

    return JSONResponse(response)


@app.get("/assets-list")
async def list_assets():
    """
    List all available assets in the assets folder, sorted by creation date (newest first).

    Returns:
        JSON response containing:
        - files: List of file objects with name, size_bytes, url, and created_at
        - total_files: Total number of files
        - total_size_bytes: Total size of all files
    """
    if not os.path.exists(ASSETS_DIR):
        return JSONResponse({"files": [], "total_files": 0, "total_size_bytes": 0})

    files = []
    total_size = 0

    try:
        import json
        from datetime import datetime

        for filename in os.listdir(ASSETS_DIR):
            # Skip metadata files
            if filename.endswith(".metadata.json"):
                continue

            filepath = os.path.join(ASSETS_DIR, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)

                # Try to load metadata
                created_at = None
                metadata_path = os.path.join(ASSETS_DIR, f"{filename}.metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            created_at = metadata.get("created_at")
                    except:
                        created_at = None

                # Fallback to file modification time if metadata not available
                if not created_at:
                    created_at = datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).isoformat()

                files.append(
                    {
                        "name": filename,
                        "size_bytes": size,
                        "url": f"/assets/{filename}",
                        "created_at": created_at,
                    }
                )
                total_size += size

        # Sort by creation date (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)

    except Exception as e:
        print(f"[API] Error listing assets: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list assets: {str(e)}"},
        )

    return JSONResponse(
        {
            "files": files,
            "total_files": len(files),
            "total_size_bytes": total_size,
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
