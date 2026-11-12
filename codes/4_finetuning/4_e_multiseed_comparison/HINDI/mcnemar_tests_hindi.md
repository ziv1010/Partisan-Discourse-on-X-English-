# McNemar Tests: Hindi Stance Evaluation

Pairwise McNemar exact tests with Holm correction.
Gold labels: `annotator_1_stance` from `hindi_stance_annotation_100.csv` (n=100)

**Note:** `gemini_zeroshot_hi_run3` uses the CSV version (16 predictions differ from the XLSX).

## Accuracy Summary

| Model | Accuracy | n |
| --- | --- | --- |
| chatgpt_fewshot_hi_run1 | 68.00% | 100 |
| chatgpt_fewshot_hi_run2 | 61.00% | 100 |
| chatgpt_fewshot_hi_run3 | 66.00% | 100 |
| chatgpt_zeroshot_hi_run1 | 71.00% | 100 |
| chatgpt_zeroshot_hi_run2 | 66.00% | 100 |
| chatgpt_zeroshot_hi_run3 | 69.00% | 100 |
| gemini_fewshot_hi_run1 | 81.00% | 100 |
| gemini_fewshot_hi_run2 | 70.00% | 100 |
| gemini_fewshot_hi_run3 | 66.00% | 100 |
| gemini_zeroshot_hi_run1 | 69.00% | 100 |
| gemini_zeroshot_hi_run2 | 65.00% | 100 |
| gemini_zeroshot_hi_run3 | 68.00% | 100 |
| mistral_lora_hi_seed100 | 78.00% | 100 |
| mistral_lora_hi_seed13 | 78.00% | 100 |
| mistral_lora_hi_seed42 | 80.00% | 100 |

## Pairwise McNemar Tests

| model_a | model_b | n | a\_right\_b\_wrong | a\_wrong\_b\_right | acc\_a | acc\_b | diff | better | exact\_p | holm\_p | sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chatgpt_fewshot_hi_run1 | chatgpt_fewshot_hi_run2 | 100 | 10 | 3 | 0.6800 | 0.6100 | +0.0700 | chatgpt_fewshot_hi_run1 | 9.2285e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | chatgpt_fewshot_hi_run3 | 100 | 4 | 2 | 0.6800 | 0.6600 | +0.0200 | chatgpt_fewshot_hi_run1 | 6.8750e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | chatgpt_zeroshot_hi_run1 | 100 | 2 | 5 | 0.6800 | 0.7100 | -0.0300 | chatgpt_zeroshot_hi_run1 | 4.5312e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | chatgpt_zeroshot_hi_run2 | 100 | 6 | 4 | 0.6800 | 0.6600 | +0.0200 | chatgpt_fewshot_hi_run1 | 7.5391e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | chatgpt_zeroshot_hi_run3 | 100 | 5 | 6 | 0.6800 | 0.6900 | -0.0100 | chatgpt_zeroshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | gemini_fewshot_hi_run1 | 100 | 7 | 20 | 0.6800 | 0.8100 | -0.1300 | gemini_fewshot_hi_run1 | 1.9157e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | gemini_fewshot_hi_run2 | 100 | 10 | 12 | 0.6800 | 0.7000 | -0.0200 | gemini_fewshot_hi_run2 | 8.3181e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | gemini_fewshot_hi_run3 | 100 | 12 | 10 | 0.6800 | 0.6600 | +0.0200 | chatgpt_fewshot_hi_run1 | 8.3181e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | gemini_zeroshot_hi_run1 | 100 | 10 | 11 | 0.6800 | 0.6900 | -0.0100 | gemini_zeroshot_hi_run1 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | gemini_zeroshot_hi_run2 | 100 | 15 | 12 | 0.6800 | 0.6500 | +0.0300 | chatgpt_fewshot_hi_run1 | 7.0111e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | gemini_zeroshot_hi_run3 | 100 | 10 | 10 | 0.6800 | 0.6800 | +0.0000 | chatgpt_fewshot_hi_run1 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | mistral_lora_hi_seed100 | 100 | 9 | 19 | 0.6800 | 0.7800 | -0.1000 | mistral_lora_hi_seed100 | 8.7159e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | mistral_lora_hi_seed13 | 100 | 8 | 18 | 0.6800 | 0.7800 | -0.1000 | mistral_lora_hi_seed13 | 7.5519e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run1 | mistral_lora_hi_seed42 | 100 | 9 | 21 | 0.6800 | 0.8000 | -0.1200 | mistral_lora_hi_seed42 | 4.2774e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | chatgpt_fewshot_hi_run3 | 100 | 3 | 8 | 0.6100 | 0.6600 | -0.0500 | chatgpt_fewshot_hi_run3 | 2.2656e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | chatgpt_zeroshot_hi_run1 | 100 | 4 | 14 | 0.6100 | 0.7100 | -0.1000 | chatgpt_zeroshot_hi_run1 | 3.0884e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | chatgpt_zeroshot_hi_run2 | 100 | 4 | 9 | 0.6100 | 0.6600 | -0.0500 | chatgpt_zeroshot_hi_run2 | 2.6685e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | chatgpt_zeroshot_hi_run3 | 100 | 4 | 12 | 0.6100 | 0.6900 | -0.0800 | chatgpt_zeroshot_hi_run3 | 7.6813e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | gemini_fewshot_hi_run1 | 100 | 6 | 26 | 0.6100 | 0.8100 | -0.2000 | gemini_fewshot_hi_run1 | 5.3505e-04 | 5.6181e-02 | No |
| chatgpt_fewshot_hi_run2 | gemini_fewshot_hi_run2 | 100 | 8 | 17 | 0.6100 | 0.7000 | -0.0900 | gemini_fewshot_hi_run2 | 1.0775e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | gemini_fewshot_hi_run3 | 100 | 11 | 16 | 0.6100 | 0.6600 | -0.0500 | gemini_fewshot_hi_run3 | 4.4207e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | gemini_zeroshot_hi_run1 | 100 | 8 | 16 | 0.6100 | 0.6900 | -0.0800 | gemini_zeroshot_hi_run1 | 1.5159e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | gemini_zeroshot_hi_run2 | 100 | 11 | 15 | 0.6100 | 0.6500 | -0.0400 | gemini_zeroshot_hi_run2 | 5.5720e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | gemini_zeroshot_hi_run3 | 100 | 9 | 16 | 0.6100 | 0.6800 | -0.0700 | gemini_zeroshot_hi_run3 | 2.2952e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run2 | mistral_lora_hi_seed100 | 100 | 8 | 25 | 0.6100 | 0.7800 | -0.1700 | mistral_lora_hi_seed100 | 4.5514e-03 | 4.5514e-01 | No |
| chatgpt_fewshot_hi_run2 | mistral_lora_hi_seed13 | 100 | 7 | 24 | 0.6100 | 0.7800 | -0.1700 | mistral_lora_hi_seed13 | 3.3269e-03 | 3.4267e-01 | No |
| chatgpt_fewshot_hi_run2 | mistral_lora_hi_seed42 | 100 | 9 | 28 | 0.6100 | 0.8000 | -0.1900 | mistral_lora_hi_seed42 | 2.5632e-03 | 2.6657e-01 | No |
| chatgpt_fewshot_hi_run3 | chatgpt_zeroshot_hi_run1 | 100 | 4 | 9 | 0.6600 | 0.7100 | -0.0500 | chatgpt_zeroshot_hi_run1 | 2.6685e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | chatgpt_zeroshot_hi_run2 | 100 | 4 | 4 | 0.6600 | 0.6600 | +0.0000 | chatgpt_fewshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | chatgpt_zeroshot_hi_run3 | 100 | 3 | 6 | 0.6600 | 0.6900 | -0.0300 | chatgpt_zeroshot_hi_run3 | 5.0781e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | gemini_fewshot_hi_run1 | 100 | 6 | 21 | 0.6600 | 0.8100 | -0.1500 | gemini_fewshot_hi_run1 | 5.9246e-03 | 5.8654e-01 | No |
| chatgpt_fewshot_hi_run3 | gemini_fewshot_hi_run2 | 100 | 9 | 13 | 0.6600 | 0.7000 | -0.0400 | gemini_fewshot_hi_run2 | 5.2347e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | gemini_fewshot_hi_run3 | 100 | 10 | 10 | 0.6600 | 0.6600 | +0.0000 | chatgpt_fewshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | gemini_zeroshot_hi_run1 | 100 | 8 | 11 | 0.6600 | 0.6900 | -0.0300 | gemini_zeroshot_hi_run1 | 6.4761e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | gemini_zeroshot_hi_run2 | 100 | 13 | 12 | 0.6600 | 0.6500 | +0.0100 | chatgpt_fewshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | gemini_zeroshot_hi_run3 | 100 | 8 | 10 | 0.6600 | 0.6800 | -0.0200 | gemini_zeroshot_hi_run3 | 8.1453e-01 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | mistral_lora_hi_seed100 | 100 | 8 | 20 | 0.6600 | 0.7800 | -0.1200 | mistral_lora_hi_seed100 | 3.5698e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | mistral_lora_hi_seed13 | 100 | 7 | 19 | 0.6600 | 0.7800 | -0.1200 | mistral_lora_hi_seed13 | 2.8959e-02 | 1.0000e+00 | No |
| chatgpt_fewshot_hi_run3 | mistral_lora_hi_seed42 | 100 | 9 | 23 | 0.6600 | 0.8000 | -0.1400 | mistral_lora_hi_seed42 | 2.0062e-02 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | chatgpt_zeroshot_hi_run2 | 100 | 8 | 3 | 0.7100 | 0.6600 | +0.0500 | chatgpt_zeroshot_hi_run1 | 2.2656e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | chatgpt_zeroshot_hi_run3 | 100 | 5 | 3 | 0.7100 | 0.6900 | +0.0200 | chatgpt_zeroshot_hi_run1 | 7.2656e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | gemini_fewshot_hi_run1 | 100 | 6 | 16 | 0.7100 | 0.8100 | -0.1000 | gemini_fewshot_hi_run1 | 5.2479e-02 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | gemini_fewshot_hi_run2 | 100 | 10 | 9 | 0.7100 | 0.7000 | +0.0100 | chatgpt_zeroshot_hi_run1 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | gemini_fewshot_hi_run3 | 100 | 13 | 8 | 0.7100 | 0.6600 | +0.0500 | chatgpt_zeroshot_hi_run1 | 3.8331e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | gemini_zeroshot_hi_run1 | 100 | 11 | 9 | 0.7100 | 0.6900 | +0.0200 | chatgpt_zeroshot_hi_run1 | 8.2380e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | gemini_zeroshot_hi_run2 | 100 | 15 | 9 | 0.7100 | 0.6500 | +0.0600 | chatgpt_zeroshot_hi_run1 | 3.0746e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | gemini_zeroshot_hi_run3 | 100 | 11 | 8 | 0.7100 | 0.6800 | +0.0300 | chatgpt_zeroshot_hi_run1 | 6.4761e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | mistral_lora_hi_seed100 | 100 | 10 | 17 | 0.7100 | 0.7800 | -0.0700 | mistral_lora_hi_seed100 | 2.4779e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | mistral_lora_hi_seed13 | 100 | 9 | 16 | 0.7100 | 0.7800 | -0.0700 | mistral_lora_hi_seed13 | 2.2952e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run1 | mistral_lora_hi_seed42 | 100 | 10 | 19 | 0.7100 | 0.8000 | -0.0900 | mistral_lora_hi_seed42 | 1.3605e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | chatgpt_zeroshot_hi_run3 | 100 | 2 | 5 | 0.6600 | 0.6900 | -0.0300 | chatgpt_zeroshot_hi_run3 | 4.5312e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | gemini_fewshot_hi_run1 | 100 | 5 | 20 | 0.6600 | 0.8100 | -0.1500 | gemini_fewshot_hi_run1 | 4.0773e-03 | 4.1181e-01 | No |
| chatgpt_zeroshot_hi_run2 | gemini_fewshot_hi_run2 | 100 | 9 | 13 | 0.6600 | 0.7000 | -0.0400 | gemini_fewshot_hi_run2 | 5.2347e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | gemini_fewshot_hi_run3 | 100 | 11 | 11 | 0.6600 | 0.6600 | +0.0000 | chatgpt_zeroshot_hi_run2 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | gemini_zeroshot_hi_run1 | 100 | 7 | 10 | 0.6600 | 0.6900 | -0.0300 | gemini_zeroshot_hi_run1 | 6.2906e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | gemini_zeroshot_hi_run2 | 100 | 11 | 10 | 0.6600 | 0.6500 | +0.0100 | chatgpt_zeroshot_hi_run2 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | gemini_zeroshot_hi_run3 | 100 | 7 | 9 | 0.6600 | 0.6800 | -0.0200 | gemini_zeroshot_hi_run3 | 8.0362e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | mistral_lora_hi_seed100 | 100 | 7 | 19 | 0.6600 | 0.7800 | -0.1200 | mistral_lora_hi_seed100 | 2.8959e-02 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | mistral_lora_hi_seed13 | 100 | 7 | 19 | 0.6600 | 0.7800 | -0.1200 | mistral_lora_hi_seed13 | 2.8959e-02 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run2 | mistral_lora_hi_seed42 | 100 | 8 | 22 | 0.6600 | 0.8000 | -0.1400 | mistral_lora_hi_seed42 | 1.6125e-02 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | gemini_fewshot_hi_run1 | 100 | 5 | 17 | 0.6900 | 0.8100 | -0.1200 | gemini_fewshot_hi_run1 | 1.6901e-02 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | gemini_fewshot_hi_run2 | 100 | 11 | 12 | 0.6900 | 0.7000 | -0.0100 | gemini_fewshot_hi_run2 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | gemini_fewshot_hi_run3 | 100 | 14 | 11 | 0.6900 | 0.6600 | +0.0300 | chatgpt_zeroshot_hi_run3 | 6.9004e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | gemini_zeroshot_hi_run1 | 100 | 10 | 10 | 0.6900 | 0.6900 | +0.0000 | chatgpt_zeroshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | gemini_zeroshot_hi_run2 | 100 | 13 | 9 | 0.6900 | 0.6500 | +0.0400 | chatgpt_zeroshot_hi_run3 | 5.2347e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | gemini_zeroshot_hi_run3 | 100 | 9 | 8 | 0.6900 | 0.6800 | +0.0100 | chatgpt_zeroshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | mistral_lora_hi_seed100 | 100 | 9 | 18 | 0.6900 | 0.7800 | -0.0900 | mistral_lora_hi_seed100 | 1.2208e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | mistral_lora_hi_seed13 | 100 | 8 | 17 | 0.6900 | 0.7800 | -0.0900 | mistral_lora_hi_seed13 | 1.0775e-01 | 1.0000e+00 | No |
| chatgpt_zeroshot_hi_run3 | mistral_lora_hi_seed42 | 100 | 10 | 21 | 0.6900 | 0.8000 | -0.1100 | mistral_lora_hi_seed42 | 7.0756e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | gemini_fewshot_hi_run2 | 100 | 17 | 6 | 0.8100 | 0.7000 | +0.1100 | gemini_fewshot_hi_run1 | 3.4690e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | gemini_fewshot_hi_run3 | 100 | 23 | 8 | 0.8100 | 0.6600 | +0.1500 | gemini_fewshot_hi_run1 | 1.0674e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | gemini_zeroshot_hi_run1 | 100 | 19 | 7 | 0.8100 | 0.6900 | +0.1200 | gemini_fewshot_hi_run1 | 2.8959e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | gemini_zeroshot_hi_run2 | 100 | 22 | 6 | 0.8100 | 0.6500 | +0.1600 | gemini_fewshot_hi_run1 | 3.7192e-03 | 3.7935e-01 | No |
| gemini_fewshot_hi_run1 | gemini_zeroshot_hi_run3 | 100 | 20 | 7 | 0.8100 | 0.6800 | +0.1300 | gemini_fewshot_hi_run1 | 1.9157e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | mistral_lora_hi_seed100 | 100 | 14 | 11 | 0.8100 | 0.7800 | +0.0300 | gemini_fewshot_hi_run1 | 6.9004e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | mistral_lora_hi_seed13 | 100 | 13 | 10 | 0.8100 | 0.7800 | +0.0300 | gemini_fewshot_hi_run1 | 6.7764e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run1 | mistral_lora_hi_seed42 | 100 | 13 | 12 | 0.8100 | 0.8000 | +0.0100 | gemini_fewshot_hi_run1 | 1.0000e+00 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | gemini_fewshot_hi_run3 | 100 | 10 | 6 | 0.7000 | 0.6600 | +0.0400 | gemini_fewshot_hi_run2 | 4.5450e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | gemini_zeroshot_hi_run1 | 100 | 8 | 7 | 0.7000 | 0.6900 | +0.0100 | gemini_fewshot_hi_run2 | 1.0000e+00 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | gemini_zeroshot_hi_run2 | 100 | 9 | 4 | 0.7000 | 0.6500 | +0.0500 | gemini_fewshot_hi_run2 | 2.6685e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | gemini_zeroshot_hi_run3 | 100 | 10 | 8 | 0.7000 | 0.6800 | +0.0200 | gemini_fewshot_hi_run2 | 8.1453e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | mistral_lora_hi_seed100 | 100 | 9 | 17 | 0.7000 | 0.7800 | -0.0800 | mistral_lora_hi_seed100 | 1.6864e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | mistral_lora_hi_seed13 | 100 | 9 | 17 | 0.7000 | 0.7800 | -0.0800 | mistral_lora_hi_seed13 | 1.6864e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run2 | mistral_lora_hi_seed42 | 100 | 11 | 21 | 0.7000 | 0.8000 | -0.1000 | mistral_lora_hi_seed42 | 1.1018e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run3 | gemini_zeroshot_hi_run1 | 100 | 6 | 9 | 0.6600 | 0.6900 | -0.0300 | gemini_zeroshot_hi_run1 | 6.0724e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run3 | gemini_zeroshot_hi_run2 | 100 | 9 | 8 | 0.6600 | 0.6500 | +0.0100 | gemini_fewshot_hi_run3 | 1.0000e+00 | 1.0000e+00 | No |
| gemini_fewshot_hi_run3 | gemini_zeroshot_hi_run3 | 100 | 9 | 11 | 0.6600 | 0.6800 | -0.0200 | gemini_zeroshot_hi_run3 | 8.2380e-01 | 1.0000e+00 | No |
| gemini_fewshot_hi_run3 | mistral_lora_hi_seed100 | 100 | 8 | 20 | 0.6600 | 0.7800 | -0.1200 | mistral_lora_hi_seed100 | 3.5698e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run3 | mistral_lora_hi_seed13 | 100 | 7 | 19 | 0.6600 | 0.7800 | -0.1200 | mistral_lora_hi_seed13 | 2.8959e-02 | 1.0000e+00 | No |
| gemini_fewshot_hi_run3 | mistral_lora_hi_seed42 | 100 | 10 | 24 | 0.6600 | 0.8000 | -0.1400 | mistral_lora_hi_seed42 | 2.4307e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run1 | gemini_zeroshot_hi_run2 | 100 | 5 | 1 | 0.6900 | 0.6500 | +0.0400 | gemini_zeroshot_hi_run1 | 2.1875e-01 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run1 | gemini_zeroshot_hi_run3 | 100 | 4 | 3 | 0.6900 | 0.6800 | +0.0100 | gemini_zeroshot_hi_run1 | 1.0000e+00 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run1 | mistral_lora_hi_seed100 | 100 | 6 | 15 | 0.6900 | 0.7800 | -0.0900 | mistral_lora_hi_seed100 | 7.8354e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run1 | mistral_lora_hi_seed13 | 100 | 7 | 16 | 0.6900 | 0.7800 | -0.0900 | mistral_lora_hi_seed13 | 9.3140e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run1 | mistral_lora_hi_seed42 | 100 | 8 | 19 | 0.6900 | 0.8000 | -0.1100 | mistral_lora_hi_seed42 | 5.2239e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run2 | gemini_zeroshot_hi_run3 | 100 | 3 | 6 | 0.6500 | 0.6800 | -0.0300 | gemini_zeroshot_hi_run3 | 5.0781e-01 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run2 | mistral_lora_hi_seed100 | 100 | 6 | 19 | 0.6500 | 0.7800 | -0.1300 | mistral_lora_hi_seed100 | 1.4633e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run2 | mistral_lora_hi_seed13 | 100 | 6 | 19 | 0.6500 | 0.7800 | -0.1300 | mistral_lora_hi_seed13 | 1.4633e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run2 | mistral_lora_hi_seed42 | 100 | 7 | 22 | 0.6500 | 0.8000 | -0.1500 | mistral_lora_hi_seed42 | 8.1301e-03 | 7.9675e-01 | No |
| gemini_zeroshot_hi_run3 | mistral_lora_hi_seed100 | 100 | 7 | 17 | 0.6800 | 0.7800 | -0.1000 | mistral_lora_hi_seed100 | 6.3915e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run3 | mistral_lora_hi_seed13 | 100 | 7 | 17 | 0.6800 | 0.7800 | -0.1000 | mistral_lora_hi_seed13 | 6.3915e-02 | 1.0000e+00 | No |
| gemini_zeroshot_hi_run3 | mistral_lora_hi_seed42 | 100 | 7 | 19 | 0.6800 | 0.8000 | -0.1200 | mistral_lora_hi_seed42 | 2.8959e-02 | 1.0000e+00 | No |
| mistral_lora_hi_seed100 | mistral_lora_hi_seed13 | 100 | 6 | 6 | 0.7800 | 0.7800 | +0.0000 | mistral_lora_hi_seed100 | 1.0000e+00 | 1.0000e+00 | No |
| mistral_lora_hi_seed100 | mistral_lora_hi_seed42 | 100 | 6 | 8 | 0.7800 | 0.8000 | -0.0200 | mistral_lora_hi_seed42 | 7.9053e-01 | 1.0000e+00 | No |
| mistral_lora_hi_seed13 | mistral_lora_hi_seed42 | 100 | 3 | 5 | 0.7800 | 0.8000 | -0.0200 | mistral_lora_hi_seed42 | 7.2656e-01 | 1.0000e+00 | No |
