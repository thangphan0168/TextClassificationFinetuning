# Experiments

## IMDB Review Dataset (Binary Classification)

**Dataset**: [Stanford IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) — 25k train / 25k test, binary sentiment (pos/neg)  
**Backbone**: `distilbert-base-uncased`  
**Epochs**: 2
**Batch size**: 16
**Eval every**: 500 steps

Two initialization strategies are compared, each swept across the same LR combinations:

- **From pretrained**: standard fine-tuning initialized directly from the HuggingFace pretrained checkpoint, all parameters unfrozen from step 1.
- **From warmed up**: the classifier head was first trained with the backbone frozen for 2 epochs (with learning rate 1e-4); the resulting checkpoint is then used as the starting point for full fine-tuning (backbone + head, all parameters unfrozen). This checkpoint achieved an accuracy of 0.7728.

| Init | Backbone LR | Head LR | Run 1 Accuracy | Run 2 Accuracy | Run 3 Accuracy | Average |
|---|---|---|---|---|---|---|
| Pretrained | 1e-4 | 1e-4 | 0.9234     | 0.9239     | 0.9235     | 0.9236 |
| Pretrained | 5e-5 | 1e-4 | 0.9277     | 0.9296     | 0.9268     | 0.9280 |
| Pretrained | 2e-5 | 1e-4 | **0.9304** | 0.9301     | 0.9311     | **0.9305** |
| Pretrained | 1e-5 | 1e-4 | 0.9271     | 0.9269     | 0.9276     | 0.9272 |
| Pretrained | 5e-5 | 5e-5 | 0.9282     | 0.9279     | 0.9285     | 0.9282 |
| Pretrained | 2e-5 | 5e-5 | 0.9302     | **0.9304** | 0.9308     | **0.9305** |
| Pretrained | 1e-5 | 5e-5 | 0.9282     | 0.9272     | 0.9286     | 0.9280 |
| Pretrained | 2e-5 | 2e-5 | 0.9298     | 0.9302     | **0.9314** | **0.9305** |
| From warmed up | 1e-4 | 1e-4 | 0.9253     | 0.9269     | 0.9192     | 0.9247 |
| From warmed up | 5e-5 | 1e-4 | 0.9311     | 0.9313     | 0.9319     | 0.9314 |
| From warmed up | 2e-5 | 1e-4 | 0.9307     | **0.9314** | **0.9321** | 0.9314 |
| From warmed up | 1e-5 | 1e-4 | 0.9277     | 0.9277     | 0.9278     | 0.9277 |
| From warmed up | 5e-5 | 5e-5 | **0.9321** | 0.9305     | 0.9308     | 0.9311 |
| From warmed up | 2e-5 | 5e-5 | 0.9316     | **0.9314** | 0.9316     | 0.9315 |
| From warmed up | 1e-5 | 5e-5 | 0.9278     | 0.9278     | 0.9280     | 0.9278 |
| From warmed up | 2e-5 | 2e-5 | 0.9312     | 0.9317     | 0.9320     | **0.9316** |

### Observations

- Warm-start initialization generally outperforms cold-start, almost every "from warmed up" run beats its pretrained counterpart at the same LR, suggesting that pre-adapting the head before unfreezing the backbone provides a better starting point for full fine-tuning.
- Backbone LR is the more sensitive hyperparameter, head LR has less effect on the results. Holding backbone LR fixed and varying head LR across 1e-4/5e-5/2e-5 changes average accuracy by no more than ~0.0005 in either init strategy — smaller than the spread between individual runs of the same config. There's no consistent evidence that different learning rates (head > backbone) outperform a matched (head = backbone) one.
- Extreme backbone LRs hurt in both strategies. 1e-5 is too conservative and limits how much the backbone adapts; 1e-4 is too aggressive and destabilises the pretrained weights early, as shown with the worst peformance across runs when backbone LR = head LR = 1e-4
- For pretrained model, backbone LR at 2e-5 performs the best where as for warmed up initialization, the best backbone LR are 2e-5 and 5e-5, suggesting the warm-up phase makes the backbone more tolerant of a slightly larger LR range during full fine-tuning.
- All 48 runs land in a narrow 0.919–0.932 accuracy band, so while the effect of warmed up initialization and backbone LR is consistent, it is also quite small on this dataset.
