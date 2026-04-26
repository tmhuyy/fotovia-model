import torch
import matplotlib.pyplot as plt
import pandas as pd # Thêm thư viện này để lưu CSV
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src.config import PROJECT_ROOT, OUTPUT_DIR
from src.dataset import build_dataloaders
from src.model import build_model

# Đường dẫn các file đầu ra
CHECKPOINT_PATH = PROJECT_ROOT / "resnext_best.pth"
IMG_PATH = OUTPUT_DIR / "resnext_confusion_matrix.png"
TXT_PATH = OUTPUT_DIR / "classification_report.txt"
CSV_PATH = OUTPUT_DIR / "classification_report.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Start script")
full_dataset, _, test_loader = build_dataloaders()
class_names = full_dataset.classes

model = build_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []

print("2. Inference starting...")
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        if i % 100 == 0:
            print(f"Processed batch {i}/{len(test_loader)}")

# 1. Tạo thư mục chứa kết quả
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Lưu Classification Report ra file .txt (Để đọc trực tiếp)
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
with open(TXT_PATH, "w") as f:
    f.write(report_dict)
print(f"\n- Đã lưu file TEXT tại: {TXT_PATH}")

# 3. Lưu Classification Report ra file .csv (Để mở bằng Excel)
report_csv = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df = pd.DataFrame(report_csv).transpose()
df.to_csv(CSV_PATH)
print(f"- Đã lưu file CSV tại: {CSV_PATH}")

# 4. Vẽ và lưu Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True)
plt.savefig(IMG_PATH, dpi=300, bbox_inches="tight")
print(f"- Đã lưu file ẢNH tại: {IMG_PATH}")

print("\n--- Xong! Bạn có thể download 3 file trên trong thư mục outputs_notebook ---")