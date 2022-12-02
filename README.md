# KoBART_Dialogue_Generator

## Setting
```
git clone https://github.com/BM-K/Dialogue-Generation-Model-Evaluation.git
pip install -r requirements.txt
```

- 데이터는 *query \t response* 형식의 tsv 파일

## How to Use (shell script example)

### Train
 
```shell
python DialogueGenerator/train.py --train-path DialogueGenerator/data/train.tsv \
	--valid-path DialogueGenerator/data/valid.tsv --device cuda:1 \
	--validation-interval 50 --max-len 32 --batch-size 128 \
	--train-log-interval 50
```

### Inference
```shell
python DialogueGenerator/inference.py --saved-model-path loss.2.9587.pth \
	--test-path DialogueGenerator/data/test.tsv --max-len 32
```

### Evaluation
```shell
python DialogueGenerator/Dialogue-Generation-Model-Evaluation/evaluation.py \
	--reference_file DialogueGenerator/inference/reference.txt \
	--hypothesis_file DialogueGenerator/inference/hypothesis.txt \
	--train_corpus_file _ \
	--subword_token ▁
```

## Arguments
### train
```shell
Usage: DialogueGenerator/train.py
Options:
      --model-path  Huggingface model path, default="gogamza/kobert-base-v1"
      --train-path  Path with training dataset, default=None
      --valid-path  Path with validation dataset, default=None
      --outpath     Path to save model checkpoint, default="./model"
      --device      Cuda device number, default="cuda"
      --max-len
      --seed
      --scheduler   Use optimizer scheduler, default="y", choice=["y", "n"]
      --batch-size
      --ignore_index
      --epoch
      --warm-ratio
      --learning-rate
      --grad-clip
      --train-log-interval  Steps to measure loss during training, default=100
      --validation-interval Steps to perform validation during training, default=100
```

### Inference
```shell
Usage: DialogueGenerator/inference.py
Options:
      --model-path  Huggingface model path, default="gogamza/kobert-base-v1"
      --test-path   Path with test dataset, default=None
      --outpath     Path to save inference results, default="inference"
      --saved-model-path Checkpoint file name of the trained model, required=True
      --device
      --max-len
      --seed
      --batch-size
      --ignore-index

```