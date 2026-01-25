import os
import random
import shutil

# ===== 설정 =====
en_dir = "../en_txt_mavhubert"
ko_dir = "../ko_txt_mavhubert"
out_root = "raw_data"

train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 0.1

seed = 42
random.seed(seed)

# ===== 1. 공통 파일명 수집 =====
en_files = set(f for f in os.listdir(en_dir) if f.endswith(".txt"))
ko_files = set(f for f in os.listdir(ko_dir) if f.endswith(".txt"))

common_files = sorted(en_files & ko_files)

assert len(common_files) > 0, "❌ 공통 파일이 없습니다"
print(f"✅ paired samples: {len(common_files)}")

# ===== 2. 셔플 =====
random.shuffle(common_files)

# ===== 3. split index 계산 =====
n = len(common_files)
n_train = int(n * train_ratio)
n_valid = int(n * valid_ratio)

train_files = common_files[:n_train]
valid_files = common_files[n_train:n_train + n_valid]
test_files  = common_files[n_train + n_valid:]

# ===== 4. 디렉토리 생성 =====
def make_dirs(split):
    os.makedirs(os.path.join(out_root, split, "en"), exist_ok=True)
    os.makedirs(os.path.join(out_root, split, "ko"), exist_ok=True)

for split in ["train", "valid", "test"]:
    make_dirs(split)

# ===== 5. 파일 복사 =====
def copy_pairs(file_list, split):
    for fname in file_list:
        shutil.copy(
            os.path.join(en_dir, fname),
            os.path.join(out_root, split, "en", fname)
        )
        shutil.copy(
            os.path.join(ko_dir, fname),
            os.path.join(out_root, split, "ko", fname)
        )

copy_pairs(train_files, "train")
copy_pairs(valid_files, "valid")
copy_pairs(test_files,  "test")

# ===== 6. 요약 출력 =====
print("📊 split summary")
print(f"train: {len(train_files)}")
print(f"valid: {len(valid_files)}")
print(f"test : {len(test_files)}")
