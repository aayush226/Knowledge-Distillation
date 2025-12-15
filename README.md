# Student vs. Sensei: Knowledge Distillation for Compact Transformer Models

Course project (NYU Deep Learning, Fall 2025) studying **logits-based knowledge distillation (KD)** to compress a BERT sentiment classifier while preserving accuracy.

## What this repo does
We fine-tune a **BERT-base teacher** on **SST-2 (GLUE)** and train a **4-layer BERT-style student** in two ways:
1) **Student (supervised)**: cross-entropy on gold labels  
2) **Student (KD)**: weighted mix of cross-entropy + KL-divergence on **softened teacher logits**

We evaluate across three labeled-data regimes: **100% / 10% / 1%** of the SST-2 training set.

## Key results (high level)
- Student is ~**2.07× smaller** and ~**2.92× faster** than the teacher (measured per-example latency).
- KD consistently improves the student vs. label-only training, with the **largest gains in the 1% data regime**.

(See the report PDF in this repo for full tables/figures.)

## Repo contents
- `kd_sst2_colab.ipynb` — **main Colab notebook** (end-to-end: data → train → eval → plots)
- `DL_Final_project.pdf` — final write-up
- `Final_Project_Proposal_KD.pdf` — proposal

### What you can configure in the notebook
- `teacher_name`: `bert-base-uncased`
- `max_length`: 128
- `batch_size`: 32
- `num_epochs`: 5
- `training_fraction`: `1.0`, `0.1`, or `0.01`
- KD params (only for KD runs):
  - `alpha` (KD loss weight)
  - `temperature`

## Method (KD objective)

Given teacher logits z_t and student logits z_s:

- Hard-label loss: L_CE = CE(z_s, y)
- Distillation loss: L_KD = KL(log_softmax(z_s/T), softmax(z_t/T)) * T^2
- Total: L = alpha * L_KD + (1 - alpha) * L_CE

## Metrics reported

- Accuracy
- Macro-F1
- Parameter count
- Per-example inference latency
- Compression factor (teacher params / student params)
- Speedup factor (teacher latency / student latency)

## Authors

- Aayush Shah 
- Bhuvan Chand Katakam
- Harini Vinu
