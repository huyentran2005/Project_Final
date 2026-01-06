# --- BLOCK 1: IMPORTS & SETUP ---
import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# 1. Cấu hình Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Đang chạy trên thiết bị: {device}")

# 2. Cấu hình Đường dẫn (Windows)
# Lưu ý: Dùng r"..." để tránh lỗi ký tự đặc biệt trong đường dẫn Windows
DATA_DIR = r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Desktop\ML & DL\Lab\Success Case 3\mytraindataset\images"
CSV_PATH = r"C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Desktop\ML & DL\Lab\Success Case 3\mytraindataset\styles.csv"
MODEL_PATH = "best_model.pth" # Giả sử bạn để file này cùng thư mục với code
INDEX_FILE = "feature_bank.pt" # File này sẽ được tạo ra để lưu cache

# Hàm seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# --- BLOCK 2: DATA PREPARATION ---
print("\n📂 Đang đọc và làm sạch dữ liệu...")
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')

# Tạo cột đường dẫn ảnh
# Lưu ý: Kiểm tra xem trong folder ảnh của bạn là chỉ có ID hay là tên file đầy đủ. 
# Code này giả định tên file là {id}.jpg
df['image_path'] = df['id'].apply(lambda x: os.path.join(DATA_DIR, str(x) + ".jpg"))

# Lọc ảnh tồn tại
def check_file_exists(path):
    return os.path.exists(path)

original_len = len(df)
df = df[df['image_path'].apply(check_file_exists)]
cleaned_len = len(df)

print(f"✅ Đã làm sạch: {original_len} -> {cleaned_len} ảnh hợp lệ.")

# --- BLOCK 3: DATASET CLASS ---
class FashionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) 

        if self.transform:
            image = self.transform(image)
        
        # Ở bước inference này, ta không cần label category nữa, chỉ cần ảnh
        return image, idx # Trả về idx để map ngược lại dataframe
    
# --- BLOCK 4: MODEL DEFINITION ---
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_size=512, pretrained=False): # Pretrained=False vì ta load weight riêng
        super(EmbeddingModel, self).__init__()
        # Load cấu trúc ResNet50 chuẩn
        self.backbone = models.resnet50(weights=None) 
        self.backbone.fc = nn.Linear(2048, embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Transform chuẩn
transform_config = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- BLOCK 5: LOAD MODEL & INDEXING ---

# 1. Load Model
model = EmbeddingModel(embedding_size=512, pretrained=False).to(device)

if os.path.exists(MODEL_PATH):
    print(f"📥 Đang load weights từ {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Xử lý trường hợp lưu cả optimizer state hoặc chỉ model state
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
else:
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}. Hãy đảm bảo bạn đã tải nó về!")

# 2. Tạo hoặc Load Index (Feature Bank)
full_dataset = FashionDataset(df, transform=transform_config)
full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=0) # Windows nên để num_workers=0 nếu lỗi

if os.path.exists(INDEX_FILE):
    print(f"⚡ Tìm thấy file index đã lưu ({INDEX_FILE}). Đang load...")
    feature_bank = torch.load(INDEX_FILE, map_location=device)
    print(f"✅ Đã load Index xong: {feature_bank.shape}")
else:
    print("⚠️ Không tìm thấy file index. Đang tạo mới (sẽ mất vài phút)...")
    feature_bank = []
    with torch.no_grad():
        for imgs, _ in tqdm(full_loader, desc="Indexing"):
            imgs = imgs.to(device)
            embeds = model(imgs)
            feature_bank.append(embeds.cpu())
    
    feature_bank = torch.cat(feature_bank, dim=0)
    torch.save(feature_bank, INDEX_FILE) # Lưu lại cho lần sau
    print(f"💾 Đã lưu Index vào {INDEX_FILE}. Lần sau sẽ không phải chạy lại bước này!")

# --- BLOCK 6: COMPUTER VISION UTILS ---

# A. Mask and Crop (Từ Case 1)
def mask_and_crop_by_background(path, output_size=(256, 256)):
    img = cv2.imread(path)
    if img is None: return None

    # 1. Denoise
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    # 2. HSV & Mask
    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    lower_background = np.array([0, 0, 200])
    upper_background = np.array([180, 50, 255])
    mask_background = cv2.inRange(hsv, lower_background, upper_background)
    mask_foreground = cv2.bitwise_not(mask_background)
    
    # 3. Morphological
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened = cv2.morphologyEx(mask_foreground, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 4. Contours
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None
    
    # 5. Crop Largest Contour
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    crop_img = bilateral[y:y+h, x:x+w]
    
    if crop_img.size == 0: return None
    return crop_img

# B. HSV Color Ranges (Từ Case 3)
color_ranges = {
    "black":  [(0, 0, 0), (180, 255, 30)],
    "white":  [(0, 0, 200), (180, 30, 255)],
    "grey":   [(0, 0, 30), (180, 50, 200)],
    "red1":   [(0, 50, 50), (10, 255, 255)],
    "red2":   [(170, 50, 50), (180, 255, 255)],
    "orange": [(11, 50, 50), (25, 255, 255)],
    "yellow": [(26, 50, 50), (35, 255, 255)],
    "green":  [(36, 50, 50), (85, 255, 255)],
    "blue":   [(86, 50, 50), (125, 255, 255)],
    "purple": [(126, 50, 50), (145, 255, 255)],
    "pink":   [(146, 50, 50), (169, 255, 255)]
}

def get_color_presence(img, color_name):
    if img is None: return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if color_name == "red":
        mask1 = cv2.inRange(hsv, np.array(color_ranges["red1"][0]), np.array(color_ranges["red1"][1]))
        mask2 = cv2.inRange(hsv, np.array(color_ranges["red2"][0]), np.array(color_ranges["red2"][1]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        if color_name not in color_ranges: return 0.0
        lower = np.array(color_ranges[color_name][0])
        upper = np.array(color_ranges[color_name][1])
        mask = cv2.inRange(hsv, lower, upper)
        
    ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
    return ratio

# --- BLOCK 7: HYBRID SEARCH LOGIC ---

def search_fashion_consultant(query_img_path, target_color, model, feature_bank, dataset, top_k=5, search_pool=150):
    # 1. Semantic Search (Shape)
    try:
        pil_img = Image.open(query_img_path).convert("RGB")
        query_tensor = transform_config(pil_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Lỗi ảnh query: {e}")
        return []

    with torch.no_grad():
        query_feat = model(query_tensor)
        
    # Tính similarity
    similarity = torch.matmul(feature_bank.to(device), query_feat.t()).squeeze()
    scores, indices = torch.topk(similarity, k=search_pool)
    
    scores = scores.cpu().numpy()
    indices = indices.cpu().numpy()
    
    # 2. Color Filtering
    print(f"🔍 Found {search_pool} items with similar shape. Filtering for '{target_color}'...")
    filtered_results = []
    
    for rank, idx in enumerate(tqdm(indices, desc="Color Filtering")): # Thêm thanh progress cho xịn
        img_path = dataset.df.iloc[idx]['image_path']
        
        # Crop để check màu chuẩn hơn
        cropped_img = mask_and_crop_by_background(img_path)
        if cropped_img is None: # Fallback nếu crop lỗi
             full_img = cv2.imread(img_path)
             if full_img is not None: cropped_img = cv2.resize(full_img, (256, 256))
        
        if cropped_img is not None:
            color_score = get_color_presence(cropped_img, target_color)
            
            # Chỉ lấy nếu màu xuất hiện > 5%
            if color_score > 0.05: 
                # Weight: 70% Color, 30% Shape (vì user request màu cụ thể)
                final_score = 0.7 * color_score + 0.3 * scores[rank]
                filtered_results.append({
                    "path": img_path,
                    "score": final_score,
                    "shape_score": scores[rank],
                    "color_score": color_score
                })

    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    return filtered_results[:top_k]

def visualize_results(query_path, target_color, results):
    if not results:
        print("❌ Không tìm thấy sản phẩm nào phù hợp!")
        return

    plt.figure(figsize=(15, 5))
    
    # Ảnh gốc
    ax = plt.subplot(1, len(results) + 1, 1)
    ax.imshow(Image.open(query_path))
    ax.set_title(f"QUERY\n(Need {target_color})", color='blue')
    ax.axis('off')
    
    # Kết quả
    for i, item in enumerate(results):
        ax = plt.subplot(1, len(results) + 1, i + 2)
        ax.imshow(Image.open(item['path']))
        ax.set_title(f"Score: {item['score']:.2f}\nColor: {item['color_score']:.2f}", color='green')
        ax.axis('off')
        
    plt.show()

# --- BLOCK 8: RUN DEMO ---
if __name__ == "__main__":
    # Lấy ngẫu nhiên 1 ảnh để test
    random_idx = random.randint(0, len(full_dataset)-1)
    query_path = full_dataset.df.iloc[random_idx]['image_path']
    target_color = "white" # Bạn có thể đổi: 'red', 'blue', 'black', 'white'...
    
    print(f"\n🤖 Demo: Tìm áo giống hình nhưng màu '{target_color}'")
    
    results = search_fashion_consultant(
        query_img_path=query_path,
        target_color=target_color,
        model=model,
        feature_bank=feature_bank,
        dataset=full_dataset,
        top_k=5
    )
    
    visualize_results(query_path, target_color, results)