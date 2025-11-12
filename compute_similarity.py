import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

warnings.filterwarnings('ignore')


class ImageEmbeddingExtractor:
    """
    Class để extract embeddings từ ảnh sử dụng Vision Foundation Models
    
    Model Comparison:
    - DINOv2-base: 86M params, 768-dim embeddings, balanced option ✅
    - DINOv2-large: ~300M params, 1024-dim embeddings, tốt nhất về accuracy nhưng chậm hơn
    - DINOv3 ConvNeXt-Tiny: 29M params, 384-dim embeddings, nhanh hơn, tốt cho production
    - SigLIP2-base: 86M params, 768-dim embeddings, tốt cho image-text và image-only tasks
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        """
        Khởi tạo model embedding
        
        Args:
            model_name: Tên model từ HuggingFace
                       - DINOv2-base (recommended): facebook/dinov2-base ✅
                       - DINOv2-large: facebook/dinov2-large
                       - DINOv3: facebook/dinov3-convnext-tiny-pretrain-lvd1689m
                       - SigLIP2-base: google/siglip2-base-patch16-224
                       - SigLIP2-large: google/siglip2-large-patch16-384
            device: Device để chạy ('cuda', 'mps', 'cpu')
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        # Load processor và model
        # SigLIP models sử dụng AutoProcessor, DINO models sử dụng AutoImageProcessor
        try:
            # Thử AutoProcessor trước (cho SigLIP)
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            # Fallback về AutoImageProcessor (cho DINO)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract embedding từ một ảnh
        
        Args:
            image: PIL Image
            
        Returns:
            embedding: Tensor shape [1, dim]
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            # SigLIP2 là vision-language model, nhưng chúng ta chỉ cần vision
            # Kiểm tra nếu model có vision_model attribute (SigLIP)
            if hasattr(self.model, 'vision_model'):
                # Chỉ gọi vision_model cho SigLIP
                vision_outputs = self.model.vision_model(**inputs)
                # Lấy pooled output hoặc CLS token
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    embedding = vision_outputs.pooler_output
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    # Lấy CLS token (token đầu tiên)
                    embedding = vision_outputs.last_hidden_state[:, 0, :]
                else:
                    embedding = vision_outputs.last_hidden_state.mean(dim=1)
            else:
                # DINO models và các models khác
                outputs = self.model(**inputs)
                
                # SigLIP2 có image_embeds (pooled image embeddings)
                # DINOv3 và một số models có pooler_output
                # DINOv2 và ViT models có last_hidden_state với CLS token
                if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                    # SigLIP2: dùng image_embeds
                    embedding = outputs.image_embeds
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # DINOv3 ConvNeXt, SigLIP: dùng pooler_output
                    embedding = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # DINOv2, CLIP: lấy CLS token (token đầu tiên)
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    # Fallback: average pooling
                    embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding
    
    def extract_embeddings_from_crops(
        self, 
        image_path: str, 
        bboxes: List[List[float]]
    ) -> List[torch.Tensor]:
        """
        Extract embeddings từ nhiều crops trong một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            bboxes: List các bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            embeddings: List các embeddings, mỗi cái shape [1, dim]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        embeddings = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Crop image
            cropped = image.crop((x1, y1, x2, y2))
            
            # Extract embedding
            embedding = self.extract_embedding(cropped)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor,
        metric: str = "cosine"
    ) -> float:
        """
        Tính similarity giữa 2 embeddings
        
        Args:
            embedding1: Embedding thứ nhất [1, dim]
            embedding2: Embedding thứ hai [1, dim]
            metric: Metric để tính similarity ('cosine' hoặc 'euclidean')
            
        Returns:
            similarity: Giá trị similarity
        """
        if metric == "cosine":
            # Cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
            return similarity.item()
        elif metric == "euclidean":
            # Euclidean distance (inverse để thành similarity)
            distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
            similarity = 1.0 / (1.0 + distance.item())
            return similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_average_embedding(
        self, 
        embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Tính average embedding từ nhiều embeddings
        
        Args:
            embeddings: List các embeddings [1, dim]
            
        Returns:
            avg_embedding: Average embedding [1, dim]
        """
        if not embeddings:
            raise ValueError("Empty embeddings list")
        
        # Stack và average
        stacked = torch.stack([emb.squeeze(0) for emb in embeddings], dim=0)  # [N, dim]
        avg_embedding = stacked.mean(dim=0, keepdim=True)  # [1, dim]
        
        return avg_embedding


def visualize_results(
    reference_image_path: str,
    target_image_path: str,
    bboxes: List[List[float]],
    similarities: List[float],
    save_path: str = None,
    model_name: str = "DINOv2-base"
):
    """
    Visualize reference image, target image với bboxes, và các crops với similarity scores
    
    Args:
        reference_image_path: Đường dẫn đến reference image
        target_image_path: Đường dẫn đến target image
        bboxes: List các bounding boxes
        similarities: List các similarity scores tương ứng với mỗi bbox
        save_path: Đường dẫn để save figure (optional)
        model_name: Tên model để hiển thị trong title
    """
    # Load images
    reference_img = Image.open(reference_image_path).convert('RGB')
    target_img = Image.open(target_image_path).convert('RGB')
    
    # Tìm bbox có similarity cao nhất
    best_idx = np.argmax(similarities)
    
    # Tạo figure với layout: reference | target | crops
    num_crops = len(bboxes)
    fig = plt.figure(figsize=(20, 5))
    
    # Sử dụng GridSpec để layout linh hoạt hơn (chỉ 1 hàng)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 3 + num_crops, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Reference image
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(reference_img)
    ax_ref.set_title('Reference Image\n(balo.jpg)', fontsize=14, fontweight='bold')
    ax_ref.axis('off')
    
    # 2. Target image với bboxes
    ax_target = fig.add_subplot(gs[0, 1:3])
    ax_target.imshow(target_img)
    
    # Vẽ bounding boxes
    for idx, (bbox, sim) in enumerate(zip(bboxes, similarities)):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Màu: xanh lá cho best match, đỏ cho các bbox khác
        if idx == best_idx:
            color = 'lime'
            linewidth = 4
        else:
            color = 'red'
            linewidth = 2
        
        # Vẽ rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax_target.add_patch(rect)
        
        # Thêm label
        label = f"#{idx+1}\n{sim:.3f}"
        ax_target.text(
            x1, y1 - 10,
            label,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    ax_target.set_title(f'Target Image với Bounding Boxes\n(thieu1.jpg)\n🏆 Best Match: Object #{best_idx+1}', 
                        fontsize=14, fontweight='bold')
    ax_target.axis('off')
    
    # 3. Các crops
    for idx, (bbox, sim) in enumerate(zip(bboxes, similarities)):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Crop image
        cropped = target_img.crop((x1, y1, x2, y2))
        
        # Subplot cho crop
        ax_crop = fig.add_subplot(gs[0, 3 + idx])
        ax_crop.imshow(cropped)
        
        # Title với similarity score
        if idx == best_idx:
            title_color = 'green'
            title = f'🏆 Object #{idx+1}\n⭐ BEST MATCH\nSimilarity: {sim:.4f}'
        else:
            title_color = 'black'
            title = f'Object #{idx+1}\nSimilarity: {sim:.4f}'
        
        ax_crop.set_title(title, fontsize=11, fontweight='bold', color=title_color)
        ax_crop.axis('off')
    
    plt.suptitle(f'🔍 Object Similarity Analysis with {model_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save hoặc show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualization saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function để tính similarity
    
    Model Options:
    - facebook/dinov2-base: DINOv2-base (86M params, 768-dim) ✅ Recommended
    - facebook/dinov2-large: DINOv2-large (~300M params, 1024-dim)
    - facebook/dinov3-convnext-tiny-pretrain-lvd1689m: DINOv3 (29M params, 384-dim)
    - google/siglip2-base-patch16-224: SigLIP2-base (86M params, 768-dim)
    - google/siglip2-large-patch16-384: SigLIP2-large (larger, better accuracy)
    """
    
    # Khởi tạo extractor với DINOv2-base
    extractor = ImageEmbeddingExtractor(model_name="facebook/dinov2-base")
    
    # Đường dẫn file
    reference_image_path = "/Users/lehoangsang/aeroeys-bluemind/balo.jpg"
    target_image_path = "/Users/lehoangsang/aeroeys-bluemind/thieu1.jpg"
    
    # Bounding boxes trong thieu1.jpg
    bboxes = [
        [457.4633, 206.5763, 597.4777, 543.2648],
        [226.9921, 301.5124, 300.6635, 370.6672],
        [227.2577, 301.3276, 300.7654, 370.9780],
        [450.1539, 279.0128, 485.9545, 308.1978],
        [458.1784, 205.4225, 518.2854, 255.2431]
    ]
    
    print("\n" + "="*60)
    print("COMPUTING SIMILARITY BETWEEN OBJECTS")
    print("="*60)
    
    # Extract embedding từ reference image (balo.jpg)
    print(f"\n📌 Extracting embedding from reference image: {reference_image_path}")
    reference_image = Image.open(reference_image_path).convert('RGB')
    reference_embedding = extractor.extract_embedding(reference_image)
    print(f"   Reference embedding shape: {reference_embedding.shape}")
    
    # Extract embeddings từ các crops trong target image
    print(f"\n📌 Extracting embeddings from {len(bboxes)} objects in: {target_image_path}")
    crop_embeddings = extractor.extract_embeddings_from_crops(
        target_image_path, 
        bboxes
    )
    print(f"   Extracted {len(crop_embeddings)} crop embeddings")
    
    # Tính similarity giữa reference và từng crop
    print("\n" + "="*60)
    print("SIMILARITY SCORES (Cosine Similarity)")
    print("="*60)
    
    similarities = []
    for idx, (bbox, crop_emb) in enumerate(zip(bboxes, crop_embeddings)):
        similarity = extractor.compute_similarity(
            reference_embedding, 
            crop_emb, 
            metric="cosine"
        )
        similarities.append(similarity)
        
        x1, y1, x2, y2 = bbox
        print(f"\nObject {idx + 1}:")
        print(f"  BBox: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
        print(f"  Size: {x2-x1:.0f} x {y2-y1:.0f} pixels")
        print(f"  Similarity: {similarity:.4f}")
    
    # Tìm object có similarity cao nhất
    max_idx = np.argmax(similarities)
    print("\n" + "="*60)
    print("BEST MATCH")
    print("="*60)
    print(f"Object {max_idx + 1} has the highest similarity: {similarities[max_idx]:.4f}")
    print(f"BBox: {bboxes[max_idx]}")
    
    # Bonus: Tính average embedding của tất cả crops (cho use case sau này)
    print("\n" + "="*60)
    print("AVERAGE EMBEDDING")
    print("="*60)
    avg_embedding = extractor.compute_average_embedding(crop_embeddings)
    print(f"Average embedding shape: {avg_embedding.shape}")
    
    # Tính similarity giữa reference và average
    avg_similarity = extractor.compute_similarity(
        reference_embedding, 
        avg_embedding, 
        metric="cosine"
    )
    print(f"Similarity with average embedding: {avg_similarity:.4f}")
    
    print("\n" + "="*60)
    print("VISUALIZING RESULTS")
    print("="*60)
    
    # Visualize kết quả
    visualize_results(
        reference_image_path=reference_image_path,
        target_image_path=target_image_path,
        bboxes=bboxes,
        similarities=similarities,
        save_path="/Users/lehoangsang/aeroeys-bluemind/similarity_visualization.png",
        model_name="DINOv2-base"
    )
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()

