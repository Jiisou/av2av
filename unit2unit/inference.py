import argparse
import numpy as np
import torch

from fairseq import checkpoint_utils, utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from unit2unit.task import UTUTPretrainingTask
from util import process_units, save_unit

def load_model(model_path, src_lang, tgt_lang, use_cuda=False):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
    print(f"loaded model config: {cfg}")

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    for model in models:
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()            
        model.prepare_for_inference_(cfg)

    task.source_language = src_lang
    task.target_language = tgt_lang

    generator = task.build_generator(
        models, cfg.generation
    )

    return task, generator

def main(args):
    print("=============== Unit2Unit Inference ===============")
    use_cuda = torch.cuda.is_available() and not args.cpu

    task, generator = load_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda) # 학습된 모델 파일로부터 번역 태스트 정의아 시퀀스 생성기 (*빔 서치)를 로드함.

    # 전처리 및 입력 준비
    with open(args.in_unit_path) as f: # 입력 파일에서 공백으로 구분된 유닛 번호 int 를 읽어와 리스트로 변환
        unit = list(map(int, f.readline().strip().split()))
    unit = task.source_dictionary.encode_line( # 유닛 시퀀스를 텍스트 사전에 매핑해서 정수형 텐서로 변환하고, 끝에 EOS 토큰 추가
        " ".join(map(lambda x: str(x), process_units(unit, reduce=True))), # 연속적으로 반복되는 유닛을 하나로 합쳐 시퀀스 길이 줄이고 번역 효율 높임.
        add_if_not_exist=False,
        append_eos=True,
    ).long()
    unit = torch.cat([ # 모델 입력 시작부에 BOS 토큰과 소스 언어 토큰 추가
        unit.new([task.source_dictionary.bos()]),
        unit,
        unit.new([task.source_dictionary.index("[{}]".format(task.source_language))]) # 모델 입력 마지막에 소스 언어 토큰 추가
    ])

    # 모델 추론 :준비된 데이터를 모델의 인코더-디코더 구조에 통과시켜 타겟 언어의 유닛을 생성
    sample = {"net_input": { # 모델 입력으로 src_tokens 키에 unit 텐서를 1차원으로 변환하여 전달
        "src_tokens": torch.LongTensor(unit).view(1,-1), # fairseq 프레임워크 표준 입력 규격인 딕셔너리 형태로 데이터 포장
    }}
    sample = utils.move_to_cuda(sample) if use_cuda else sample # 참이면, 연산 가속을 위해 데이터 샘플을 gpu 메모리로 복사.

    pred = task.inference_step(
        generator, # 타겟 유닛 시퀀스를 생성하는 실제 추론 수행
        None,
        sample,
    )[0][0] # 생성된 여러 가설 중 확률이 가장 높은 첫 번째 결과를 선택 (=best beam)

    # 후처리 및 저장 :모델이 출력한 토큰 인덱스를 다시 사람이 읽거나 보코더에 입력 가능한 유닛 문자열로 복원
    pred_str = task.target_dictionary.string( # 토큰(유닛) 번호를 공백으로 구분된 문자열로 변환
        pred["tokens"].int().cpu(), # pred["tokens"]는 모델이 생성한 정수형 유닛 시퀀스 텐서
        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator) # 모델 출력에서 불필요한 토큰(BOS, EOS, 언어 토큰) 제거
    )

    save_unit(pred_str, args.out_unit_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-unit-path", type=str, required=True, help="File path of source unit input"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="File path of target unit output"
    )
    parser.add_argument(
        "--utut-path", type=str, required=True, help="path to the UTUT pre-trained model"
    )
    parser.add_argument(
        "--src-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="source language"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="target language"
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli_main()
