# TMBB LSTM

Bidirectional LSTM (PyTorch) for TMBB protein sequence generation.

---

## Install

```bash
pip install torch numpy
```

---

## Data

```
<basename>_train.fasta
<basename>_valid.fasta
```

---

## Train

Here is an example of using the script for a dataset with basename "iiab_tmbb".
```bash
python3 lstm_lm_bidirectional.py <basename> <outfolder> 10 1 1024 rmsprop 1e-4 250 \
  --lstm_num_nodes 128 \
  --lstm_frag_size 50 \
  --gpu_device 0 \
  --task training
```

---

## Inference
Here is the format for using this script for inferene (after the training run).
```bash
python3 lstm_lm_bidirectional.py <basename> <outfolder> ... \
  --task inference \
  --lstm_num_nodes <n> \
  --lstm_frag_size <n> \
  --method <greedy|pure|topk|nucleus>
```

---

## Output
For the training run, the script produces the folowing output files:
* `model_<epoch>`
* `train_evolution.tsv`
* `epoch_<epoch>.fasta`

For the inference run, the script produces the following output file:
* `valid_evolution.tsv`

