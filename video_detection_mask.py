# ========== IMPORTS ==========
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort
from pathlib import Path
from itertools import batched
from tqdm import tqdm
from ultralytics.models.yolo import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from torchvision.ops import masks_to_boxes, roi_align
from torchvision.transforms.functional import to_pil_image

# ========== CONFIGURATION ==========
MODEL_DIR = Path("models")
BATCH_SIZE = 1
REF_FOLDER = Path("public_test/samples/CardboardBox_0/object_images")                    # Folder chứa ảnh tham chiếu để huấn luyện
TARGET_IMAGE_PATH = Path("my_frames/frame_002889_01_55.jpg")      # Đường dẫn đầy đủ đến ảnh target cần tìm kiếm

# ORT Providers configuration
ORT_PROVIDERS = [
    "CUDAExecutionProvider",
    (
        "CoreMLExecutionProvider",
        {
            "ModelFormat": "MLProgram",
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "1",
        },
    ),
    "CPUExecutionProvider",
]

# Type mapping for ONNX Runtime
ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(uint16)": np.uint16,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(double)": np.float64,
    "tensor(bool)": bool,
    "tensor(float16)": np.float16,
}

# ========== HELPER FUNCTIONS ==========
def get_ort_session_device_type(session: ort.InferenceSession) -> str:
    """Get device type from ONNX Runtime session."""
    provider = session.get_providers()[0]
    return provider[: provider.index("ExecutionProvider")].lower()

# ========== STEP 1: INITIALIZE BEN2 MODEL ==========
def setup_ben2_session():
    """Initialize BEN2 ONNX session for background extraction."""
    session = ort.InferenceSession(
        # MODEL_DIR / "ben2" / "fp16.onnx", providers=ORT_PROVIDERS
        MODEL_DIR / "BEN2-folded.onnx", providers=ORT_PROVIDERS
    )
    io_binding = session.io_binding()
    input_node = session.get_inputs()[0]
    output_node = session.get_outputs()[0]

    device_type = get_ort_session_device_type(session)
    if device_type == "coreml":
        device_type = "cpu"

    b, c, h, w = input_node.shape
    input_batch = np.empty([b, c, h, w], dtype=np.float32)

    input_ortvalue = None
    if device_type != "cpu":
        input_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
            input_node.shape, ORT_TYPE_TO_NUMPY[input_node.type], device_type
        )
        io_binding.bind_ortvalue_input(input_node.name, input_ortvalue)
    else:
        io_binding.bind_cpu_input(input_node.name, input_batch)

    output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
        output_node.shape, ORT_TYPE_TO_NUMPY[output_node.type], device_type
    )
    io_binding.bind_ortvalue_output(output_node.name, output_ortvalue)

    return session, io_binding, input_batch, input_ortvalue, output_ortvalue, device_type


# ========== STEP 2: PROCESS REF IMAGES WITH BEN2 ==========
def process_ref_images_ben2(ref_paths):
    """Extract masks from reference images using BEN2 model."""
    session, io_binding, input_batch, input_ortvalue, output_ortvalue, device_type = (
        setup_ben2_session()
    )

    input_rgb_list: list[np.ndarray | None] = [None] * BATCH_SIZE

    for batch_paths in batched(tqdm(ref_paths, desc="Processing ref images"), BATCH_SIZE):
        for idx, img_path in enumerate(batch_paths):
            input_rgb_list[idx] = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if input_rgb_list[idx] is None:
                print(f"Warning: Could not read {img_path}")
                continue

            img_data = input_rgb_list[idx]
            resized_image = cv2.resize(img_data, (1024, 1024))
            chw_image = resized_image.transpose(2, 0, 1)
            np.divide(chw_image, np.iinfo(np.uint8).max, out=input_batch[idx])

        # Inference
        if device_type != "cpu" and input_ortvalue is not None:
            input_ortvalue.update_inplace(input_batch)
        session.run_with_iobinding(io_binding)
        outputs = output_ortvalue.numpy()

        # Postprocess
        for img_path, input_rgb, output in zip(batch_paths, input_rgb_list, outputs):
            if input_rgb is None:
                continue

            raw_mask = output.squeeze()
            min_val = raw_mask.min()
            max_val = raw_mask.max()

            normalized_mask = (raw_mask - min_val) / (
                max_val - min_val + np.finfo(np.float32).eps
            )
            normalized_mask *= np.iinfo(np.uint8).max

            resized_mask = cv2.resize(
                normalized_mask.astype(np.uint8),
                dsize=(input_rgb.shape[1], input_rgb.shape[0]),
            )

            # Save background mask
            save_path = img_path.with_name(f"{img_path.stem}_bg.png")
            cv2.imwrite(str(save_path), resized_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Save object with alpha channel
            bgra_image = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2BGRA)
            bgra_image[:, :, 3] = resized_mask
            save_path = img_path.with_name(f"{img_path.stem}_obj.png")
            cv2.imwrite(str(save_path), bgra_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


# ========== STEP 3: EXTRACT VPE FROM REF IMAGES ==========
def extract_vpe_embeddings(ref_paths, yoloe):
    """Extract visual prompt embeddings from reference images."""
    predictor = YOLOEVPDetectPredictor(
        overrides={"task": "detect", "mode": "predict", "batch": 1}
    )
    predictor.setup_model(yoloe.model)

    all_vpe = []

    for ref_path in tqdm(ref_paths, desc="Extracting VPE"):
        bg_path = ref_path.with_name(f"{ref_path.stem}_bg.png")
        bg_np = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
        if bg_np is None:
            print(f"Warning: Could not read {bg_path}")
            continue

        # Normalize mask to binary (0 or 1)
        bg_mask = (bg_np > (255 // 2)).astype(np.uint8)

        predictor.set_prompts(
            {
                "masks": [bg_mask],  # Keep as numpy array, not tensor
                "cls": np.array([0]),
            }
        )

        vpe = predictor.get_vpe(ref_path)
        all_vpe.append(vpe)

    avg_vpe = torch.mean(torch.stack(all_vpe), dim=0)
    return avg_vpe


# ========== STEP 4: PROCESS IMAGE WITH YOLO DETECTION ==========
def process_image_crops(yoloe, image_path):
    """Detect objects in image and extract crops using YOLO.

    Args:
        yoloe: YOLO model
        image_path: Path to input image

    Returns:
        crops: Tensor of detected object crops [N, 3, 224, 224]
        or None if no detections
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return None

    # Run YOLO detection
    results = yoloe.predict(image_path, conf=0.0001, iou=0.01)
    if not results or results[0].masks is None:
        print(f"No objects detected in {image_path.name}")
        return None

    result = results[0]

    # Convert original image to tensor
    img = torch.as_tensor(result.orig_img)
    img = img.permute(2, 0, 1)
    img = img.float() / 255.0
    H_img, W_img = img.shape[-2:]

    # Get masks from detection
    masks = torch.as_tensor(result.masks.data).float()
    N, _, _ = masks.shape

    # Resize masks to match image size
    masks = F.interpolate(
        masks.unsqueeze(1),
        size=(H_img, W_img),
        mode="nearest",
    ).squeeze(1)

    # Get bounding boxes from masks
    boxes = masks_to_boxes(masks)

    # Create masked images
    img_batch = img.unsqueeze(0).expand(N, -1, -1, -1)
    masks_batched = masks.unsqueeze(1)
    masked_imgs = img_batch * masks_batched

    # Prepare ROIs for roi_align
    batch_idx = torch.arange(N).float().unsqueeze(1)
    rois = torch.cat([batch_idx, boxes], dim=1)

    # Extract crops using roi_align
    crops = roi_align(
        masked_imgs,
        rois,
        output_size=(224, 224),
        spatial_scale=1.0,
        aligned=True,
    )

    print(f"Detected and extracted {N} object crops from {image_path.name}")
    return crops


# ========== STEP 5: COMPUTE DINO EMBEDDINGS AND SIMILARITY ==========
def compute_similarity(crops, ref_paths):
    """Compute DINOv3 embeddings and cosine similarity."""
    # Load DINOv3 model from local folder
    dinov3_path = MODEL_DIR / "dinov3"
    print(f"Loading DINOv3 from {dinov3_path}")
    model = AutoModel.from_pretrained(
        str(dinov3_path),
        device_map="auto",
    )

    dino_inputs = [load_image(to_pil_image(crop[[2, 1, 0], ...])) for crop in crops]
    processor = AutoImageProcessor.from_pretrained(str(dinov3_path))

    dino_pixels = processor(
        images=dino_inputs,
        return_tensors="pt",
        device=model.device,
    )

    ref_pixels = processor(
        images=[
            load_image(str(path.with_name(f"{path.stem}_obj.png"))) for path in ref_paths
        ],
        return_tensors="pt",
        device=model.device,
    )

    with torch.inference_mode():
        dino_input = model(**dino_pixels)
        ref_input = model(**ref_pixels)

        M = F.cosine_similarity(
            dino_input.pooler_output.unsqueeze(1),
            ref_input.pooler_output.unsqueeze(0),
            dim=-1,
        )
        M = M.mean(dim=1)

        best_idx = M.argmax()
        print(f"Best match index: {best_idx}")
        print(f"Similarity scores: {M}")

        return best_idx, M


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize YOLOE model
    model_name = MODEL_DIR / "yoloe-v8l-seg.pt"
    yoloe = YOLOE(model_name).eval()

    # Load all reference images for training
    ref_paths = sorted(REF_FOLDER.glob("*.jpg"))
    print(f"Found {len(ref_paths)} reference images in '{REF_FOLDER}':")
    for path in ref_paths:
        print(f"  - {path.name}")

    # Load target image
    if not TARGET_IMAGE_PATH.exists():
        print(f"\nError: Target image not found at '{TARGET_IMAGE_PATH}'")
        print("Please set TARGET_IMAGE_PATH correctly in configuration")
        exit(1)
    else:
        print(f"\nTarget image: {TARGET_IMAGE_PATH.name} (from {TARGET_IMAGE_PATH.parent})")

    # Step 1: Process ref images with BEN2
    print("\n=== Step 1: Processing reference images with BEN2 ===")
    process_ref_images_ben2(ref_paths)

    # Step 2: Extract VPE embeddings
    print("\n=== Step 2: Extracting VPE embeddings ===")
    avg_vpe = extract_vpe_embeddings(ref_paths, yoloe)
    yoloe.set_classes(["obj"], avg_vpe)

    # Step 3: Process target image and extract crops
    print("\n=== Step 3: Processing target image ===")
    target_crops = process_image_crops(yoloe, TARGET_IMAGE_PATH)

    if target_crops is not None:
        # Step 4: Compute similarity
        print("\n=== Step 4: Computing DINOv3 similarity ===")
        best_idx, similarity_scores = compute_similarity(target_crops, ref_paths)

        # Print results
        print("\n" + "="*60)
        print("MATCHING RESULTS")
        print("="*60)
        print(f"Target image: {TARGET_IMAGE_PATH.name}")
        print(f"Best matching reference: {ref_paths[best_idx].name}")
        print(f"Similarity score: {similarity_scores[best_idx]:.4f}")
        print(f"\nAll similarity scores:")
        for path, score in zip(ref_paths, similarity_scores):
            print(f"  {path.name}: {score:.4f}")
        print(f"\nSave best image:")
        crop = target_crops[best_idx]
        # Convert torch tensor to numpy and adjust for cv2
        crop_np = (crop.cpu().numpy() * 255).astype(np.uint8)
        crop_np = np.transpose(crop_np, (1, 2, 0))  # CHW -> HWC
        crop_np = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)  # RGB -> BGR for cv2
        cv2.imwrite("crop_image.jpg", crop_np)
        print("Saved best matching crop to 'crop_image.jpg'")
        print("="*60)
    else:
        print("Could not process target image")