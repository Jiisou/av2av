for f in /home/2022113135/datasets/a2a/split_units/test/ko/*.txt; do
    cat "$f"
    echo ""
done > raw_data/test.ko

for f in /home/2022113135/datasets/a2a/split_units/test/en/*.txt; do
    cat "$f"
    echo ""
done > raw_data/test.en

fairseq-preprocess \
    --source-lang en \
    --target-lang ko \
    --trainpref a2a/raw_data/train \
    --validpref a2a/raw_data/valid \
    --testpref a2a/raw_data/test \
    --destdir a2a/dataset_mbart_ft_bin_data/en/ko \
    --srcdict a2a/utut_pretrain/dict.txt \
    --tgtdict a2a/utut_pretrain/dict.txt \
    --workers 4