# Oral Diseases Classification

Image classification of six oral conditions using a custom CNN and transfer learning (ResNet50V2, EfficientNetB0, MobileNetV2), served through a Gradio web app.

The best model, a fine-tuned **ResNet50V2**, reaches **93.2% test accuracy** and **0.906 macro-F1** on a leakage-controlled test set.

<img width="1261" height="718" alt="Screenshot 2026-10-26 184949" src="https://github.com/user-attachments/assets/5bc6287c-9b28-450a-9901-5ff15520d0eb" />
<img width="1258" height="677" alt="Screenshot 2026-10-26 185010" src="https://github.com/user-attachments/assets/60a2b903-fbf7-4f92-9b83-175d3e9dbf66" />


## Classes

`Calculus` · `Caries` · `Gingivitis` · `Hypodontia` · `Mouth_Ulcer` · `Tooth_Discoloration`

## Dataset

[Oral Diseases Dataset](https://www.kaggle.com/datasets/miraosama/oral-diseases-dataset) on Kaggle.

The raw data had serious quality issues, so it was cleaned before training:

| Step | Result |
|------|--------|
| Original images (original folders only, augmented copies excluded) | 5,563 |
| Exact duplicates across *different* classes (label conflicts) removed | 392 groups, 1,510 images |
| Exact duplicates within the same class removed | 546 images |
| Near-duplicates grouped with MobileNetV2 embeddings (cosine similarity ≥ 0.90) | 5 cross-class images dropped |
| **Final dataset** | **3,502 images** |

The split uses `StratifiedGroupKFold`, so near-duplicate images never appear in more than one of train, validation and test. This avoids the inflated scores a plain random split would give.

| Split | Images |
|-------|--------|
| Train | 2,503 |
| Validation | 486 |
| Test | 513 |

Class imbalance is handled with balanced class weights, and the best model is selected by validation **macro-F1** instead of accuracy.

## Models and Results

All models share the same split, augmentation (flip, rotation, zoom, contrast) and class weights. The pretrained backbones were trained in two stages: a frozen-backbone phase, then partial fine-tuning.

| Model | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Params (M) | Inference (ms/img) |
|-------|--------:|-------------:|---------:|--------------:|-----------:|-------------------:|
| **ResNet50V2 (tuned)** | 0.936 | 0.879 | **0.932** | **0.906** | 23.58 | 3.66 |
| EfficientNetB0 | 0.889 | 0.828 | 0.881 | 0.845 | 4.06 | 3.16 |
| ResNet50V2 (default) | 0.893 | 0.813 | 0.905 | 0.854 | 23.58 | 4.43 |
| MobileNetV2 | 0.852 | 0.776 | 0.866 | 0.834 | 2.27 | 2.08 |
| Custom CNN | 0.702 | 0.630 | 0.630 | 0.550 | 0.46 | 1.27 |

Full numbers are in [`comparison.csv`](comparison.csv).

### Hyperparameter tuning (ResNet50V2)

Five configurations were compared on validation macro-F1 (see [`tuning_log_groupsplit.csv`](tuning_log_groupsplit.csv)). The best was dropout 0.5, fine-tuning learning rate 1e-4, and the last 100 layers unfrozen. Tuning improved test macro-F1 from 0.854 to 0.906.

### Per-class performance (best model, test set)

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| Calculus | 0.873 | 0.945 | 0.908 | 73 |
| Caries | 0.828 | 0.828 | 0.828 | 29 |
| Gingivitis | 0.947 | 0.936 | 0.942 | 173 |
| Hypodontia | 0.971 | 0.950 | 0.960 | 179 |
| Mouth_Ulcer | 1.000 | 0.875 | 0.933 | 32 |
| Tooth_Discoloration | 0.806 | 0.926 | 0.862 | 27 |

### Key takeaways

- Every pretrained model clearly beat the custom CNN, which learns from scratch on only about 2.5k training images.
- Fine-tuning about 100 layers worked best. Too few (30) limited adaptation, and too many (150) lowered validation F1.
- MobileNetV2 has about 10× fewer parameters than ResNet50V2 and runs faster, so it is a reasonable choice for mobile or edge use at a moderate accuracy cost.
- The smaller classes (Caries, Tooth_Discoloration) are the weakest, which matches their limited number of samples.

## Project Structure

```
├── app.py                        # Gradio web app
├── best_model_small.keras        # Trained model used by the app
├── class_names.json              # Class labels
├── comparison.csv                # Model comparison results
├── tuning_log_groupsplit.csv     # Hyperparameter tuning log
├── requirements.txt
└── notebook2a570f85f9.ipynb      # Data cleaning, training and evaluation
```

## Run Locally

```bash
git clone https://github.com/Gerges1245/oral-diseases-classification.git
cd oral-diseases-classification
pip install -r requirements.txt
python app.py
```

Then open the local URL printed in the terminal.

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · Gradio

## Disclaimer

This project is for educational purposes only. It is not a medical device and must not be used for diagnosis. Consult a qualified dentist for any oral health concern.

## Author

**Gerges** — Computer Science student at MSA University · [GitHub](https://github.com/Gerges1245)
