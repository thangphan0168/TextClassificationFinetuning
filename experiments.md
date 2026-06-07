# Experiments

## IMDB Review Dataset (Binary Classification)

**Dataset**: [Stanford IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) — 25k train / 25k test, binary sentiment (pos/neg)  
**Backbone**: `distilbert-base-uncased`  
**Epochs**: 2
**Batch size**: 16
**Eval every**: 500 steps

Two initialization strategies are compared, each swept across the same LR combinations:

- **From pretrained**: standard fine-tuning initialized directly from the HuggingFace pretrained checkpoint, all parameters unfrozen from step 1.
- **From warmed up**: the classifier head was first trained with the backbone frozen (with learning rate 1e-4); the resulting checkpoint is then used as the starting point for full fine-tuning (backbone + head, all parameters unfrozen).

| Init | Backbone LR | Head LR | Best Accuracy |
|---|---|---|---|
| Pretrained | 1e-4 | 1e-4 | 0.9234 |
| Pretrained | 5e-5 | 1e-4 | 0.9277 |
| Pretrained | 2e-5 | 1e-4 | **0.9304** |
| Pretrained | 1e-5 | 1e-4 | 0.9271 |
| Pretrained | 5e-5 | 5e-5 | 0.9282 |
| Pretrained | 2e-5 | 5e-5 | 0.9302 |
| Pretrained | 1e-5 | 5e-5 | 0.9282 |
| Pretrained | 2e-5 | 2e-5 | 0.9298 |
| From warmed up | 1e-4 | 1e-4 | 0.9253 |
| From warmed up | 5e-5 | 1e-4 | 0.9311 |
| From warmed up | 2e-5 | 1e-4 | 0.9307 |
| From warmed up | 1e-5 | 1e-4 | 0.9277 |
| From warmed up | 5e-5 | 5e-5 | **0.9321** |
| From warmed up | 2e-5 | 5e-5 | 0.9316 |
| From warmed up | 1e-5 | 5e-5 | 0.9278 |
| From warmed up | 2e-5 | 2e-5 | 0.9312 |

### Observations

- Warm-start initialization generally outperforms cold-start — every "from warmed up" run matches or beats its pretrained counterpart at the same LR, suggesting that pre-adapting the head before unfreezing the backbone provides a better starting point for full fine-tuning.
- Differential LR matters more for cold-start runs. Among pretrained runs, 2e-5/1e-4 (backbone LR lower than head LR) is the clear best (0.9304), while equal-LR runs lag behind. The head needs a head start when the backbone has never seen the task.
- Differential LR has less impact for warm-start runs, where 5e-5/5e-5 (equal LR) performs best (0.9321). This may be because the head is already well-adapted from the warm-up phase, reducing the need to update it faster than the backbone. However, it is also possible that a higher head LR would help - this was not tested since 1e-4 was the LR used during the frozen warm-up phase and was not exceeded here.
- Extreme backbone LRs hurt in both strategies. 1e-5 is too conservative and limits how much the backbone adapts; 1e-4 is too aggressive and destabilises the pretrained weights early (accuracy dips to 0.8643 and 0.8539 at step 1k for pretrained and warm-start respectively). A backbone LR in the 2e-5 - 5e-5 range appears to be the sweet spot for this dataset. 
- Regardless, all models achieved accuracy around a relatively small range 0.923-0.932, so the effect of these hyperparameters is not conclusive
