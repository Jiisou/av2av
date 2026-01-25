# UTUT (Unit-to-Unit Translation) Fine-tuning Guide  
**English → Korean Speech Unit Translation**

본 문서는 **UTUT 모델**을 영어–한국어 병렬 음성 데이터로 fine-tuning 하기 위한 전체 데이터 파이프라인과 실행 방법을 설명한다.  
본 실험은 **mavHuBERT 기반 AV2Unit**을 통해 음성을 discrete unit으로 변환한 뒤, **fairseq의 `translation_from_pretrained_bart` task**를 사용하여 unit-to-unit 번역을 수행한다.

---

## Overview: UTUT Workflow

```
Parallel Audio (en/*.wav, ko/*.wav)
        ↓  [AV2Unit]
Discrete Unit Files (units/en/*.txt, units/ko/*.txt)
        ↓  [Concatenate]
Raw Parallel Text (train.en, train.ko, valid.en, valid.ko)
        ↓  [fairseq-preprocess]
Binarized Dataset (*.bin, *.idx)
        ↓  [finetune_en_ko.py]
Fine-tuned UTUT Model
```

> ⚠️ 본 파이프라인은 **텍스트 번역이 아닌 speech unit-to-unit translation** 문제 설정이다.

---

## Data Preparation

### Step 1. 병렬 오디오 데이터 준비

영어–한국어 **1:1로 대응되는 병렬 음성 데이터**가 필요하다.

```
audio/
├── en/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── ko/
    ├── sample_001.wav
    ├── sample_002.wav
    └── ...
```

- 파일명은 언어 간 반드시 동일
- `(en, ko)` 오디오 쌍이 하나의 번역 샘플을 구성

---

### Step 2. AV2Unit을 이용한 Audio → Discrete Unit 변환

#### 영어 오디오

```bash
PYTHONPATH=fairseq python av2unit/inference.py     --in-vid-path audio/en/sample_001.wav     --out-unit-path units/en/sample_001.txt     --ckpt-path modelckpt/mavhubert_large_noise.pt     --modalities audio
```

#### 한국어 오디오

```bash
PYTHONPATH=fairseq python av2unit/inference.py     --in-vid-path audio/ko/sample_001.wav     --out-unit-path units/ko/sample_001.txt     --ckpt-path modelckpt/mavhubert_large_noise.pt     --modalities audio
```

---

### Step 3. Fairseq용 Raw Text 데이터 구성

```bash
mkdir -p unit2unit/utut_finetune/raw_data
```

```bash
for f in units/en/train_*.txt; do
    cat "$f"
    echo ""
done > raw_data/train.en
```

```bash
for f in units/ko/train_*.txt; do
    cat "$f"
    echo ""
done > raw_data/train.ko
```

⚠️ 각 줄은 반드시 병렬 정렬되어야 함.

---

### Step 4. fairseq-preprocess

```bash
cd unit2unit

fairseq-preprocess \
    --source-lang en \
    --target-lang ko \
    --trainpref a2a/raw_data/train \
    --validpref a2a/raw_data/valid \
    --testpref a2a/raw_data/test \
    --destdir utut_finetune/data/dataset_mbart_ft_bin_data/en/ko \
    --srcdict a2a/utut_pretrain/dict.txt \
    --tgtdict a2a/utut_pretrain/dict.txt \
    --workers 4
```

---

## Run UTUT Fine-tuning
> pretrained unit-mbart checkpoint : [Unit mBART](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/enhanced_direct_s2st_discrete_units.md#unit-mbart) --> `/../utut_finetune/`

```bash
cd unit2unit/utut_finetune

PYTHONPATH=/path/to/fairseq OMP_NUM_THREADS=1 python finetune_en_ko.py data/dataset_mbart_ft_bin_data/en/ko --arch mbart_large \
--task translation_from_pretrained_bart \
--criterion label_smoothed_cross_entropy \
--user-dir ./
```

---

## Notes

- 데이터 정렬 오류는 학습 실패의 주요 원인
- unit dictionary는 pretrain과 동일해야 함
