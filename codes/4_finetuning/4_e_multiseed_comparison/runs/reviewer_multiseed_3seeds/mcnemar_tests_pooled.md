# Pooled McNemar Tests: /scratch/ziv_baretto/Research_X/Partisan-Discourse-on-X-English-/codes/4_finetuning/4_e_multiseed_comparison/runs/reviewer_multiseed_3seeds

McNemar's paired test is computed from the two discordant cells:
`a_correct_b_wrong` and `a_wrong_b_correct`.
The reported exact p-value is the two-sided binomial McNemar p-value; `holm_p` is Holm-corrected within the table/family.

| seed_count | model_a | model_b | n_compared | a_correct_b_wrong | a_wrong_b_correct | accuracy_diff_a_minus_b | better_model | exact_p | holm_p | holm_significant_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | bert | roberta | 795 | 99 | 121 | -0.0277 | roberta | 0.1567 | 1.0000 | False |
| 3 | bert | pyabsa | 795 | 135 | 111 | 0.0302 | bert | 0.1424 | 1.0000 | False |
| 3 | bert | mistral_base | 795 | 180 | 193 | -0.0164 | mistral_base | 0.5344 | 1.0000 | False |
| 3 | bert | mistral_fewshot | 795 | 157 | 221 | -0.0805 | mistral_fewshot | 0.0012 | 0.0210 | True |
| 3 | bert | mistral_lora | 795 | 78 | 257 | -0.2252 | mistral_lora | 1.891e-23 | 6.052e-22 | True |
| 3 | bert | chatgpt_fewshot | 795 | 74 | 264 | -0.2390 | chatgpt_fewshot | 3.637e-26 | 1.309e-24 | True |
| 3 | bert | chatgpt_zeroshot | 795 | 69 | 266 | -0.2478 | chatgpt_zeroshot | 2.027e-28 | 7.905e-27 | True |
| 3 | bert | gemini_fewshot | 795 | 75 | 259 | -0.2314 | gemini_fewshot | 7.565e-25 | 2.648e-23 | True |
| 3 | bert | gemini_zeroshot | 795 | 78 | 257 | -0.2252 | gemini_zeroshot | 1.891e-23 | 6.052e-22 | True |
| 3 | roberta | pyabsa | 795 | 133 | 87 | 0.0579 | roberta | 0.0023 | 0.0398 | True |
| 3 | roberta | mistral_base | 795 | 151 | 142 | 0.0113 | roberta | 0.6403 | 1.0000 | False |
| 3 | roberta | mistral_fewshot | 795 | 133 | 175 | -0.0528 | mistral_fewshot | 0.0193 | 0.3093 | False |
| 3 | roberta | mistral_lora | 795 | 68 | 225 | -0.1975 | mistral_lora | 8.613e-21 | 2.326e-19 | True |
| 3 | roberta | chatgpt_fewshot | 795 | 71 | 239 | -0.2113 | chatgpt_fewshot | 2.040e-22 | 6.120e-21 | True |
| 3 | roberta | chatgpt_zeroshot | 795 | 66 | 241 | -0.2201 | chatgpt_zeroshot | 1.447e-24 | 4.921e-23 | True |
| 3 | roberta | gemini_fewshot | 795 | 69 | 231 | -0.2038 | gemini_fewshot | 1.390e-21 | 3.892e-20 | True |
| 3 | roberta | gemini_zeroshot | 795 | 68 | 225 | -0.1975 | gemini_zeroshot | 8.613e-21 | 2.326e-19 | True |
| 3 | pyabsa | mistral_base | 795 | 159 | 196 | -0.0465 | mistral_base | 0.0559 | 0.8384 | False |
| 3 | pyabsa | mistral_fewshot | 795 | 135 | 223 | -0.1107 | mistral_fewshot | 3.835e-06 | 7.670e-05 | True |
| 3 | pyabsa | mistral_lora | 795 | 70 | 273 | -0.2553 | mistral_lora | 1.901e-29 | 7.604e-28 | True |
| 3 | pyabsa | chatgpt_fewshot | 795 | 69 | 283 | -0.2692 | chatgpt_fewshot | 6.819e-32 | 3.001e-30 | True |
| 3 | pyabsa | chatgpt_zeroshot | 795 | 66 | 287 | -0.2780 | chatgpt_zeroshot | 5.594e-34 | 2.518e-32 | True |
| 3 | pyabsa | gemini_fewshot | 795 | 74 | 282 | -0.2616 | gemini_fewshot | 1.010e-29 | 4.140e-28 | True |
| 3 | pyabsa | gemini_zeroshot | 795 | 78 | 281 | -0.2553 | gemini_zeroshot | 4.869e-28 | 1.850e-26 | True |
| 3 | mistral_base | mistral_fewshot | 795 | 36 | 87 | -0.0642 | mistral_fewshot | 4.849e-06 | 9.213e-05 | True |
| 3 | mistral_base | mistral_lora | 795 | 31 | 197 | -0.2088 | mistral_lora | 9.828e-31 | 4.226e-29 | True |
| 3 | mistral_base | chatgpt_fewshot | 795 | 51 | 228 | -0.2226 | chatgpt_fewshot | 6.931e-28 | 2.565e-26 | True |
| 3 | mistral_base | chatgpt_zeroshot | 795 | 46 | 230 | -0.2314 | chatgpt_zeroshot | 1.341e-30 | 5.633e-29 | True |
| 3 | mistral_base | gemini_fewshot | 795 | 63 | 234 | -0.2151 | gemini_fewshot | 2.727e-24 | 8.998e-23 | True |
| 3 | mistral_base | gemini_zeroshot | 795 | 73 | 239 | -0.2088 | gemini_zeroshot | 9.526e-22 | 2.762e-20 | True |
| 3 | mistral_fewshot | mistral_lora | 795 | 36 | 151 | -0.1447 | mistral_lora | 5.950e-18 | 1.488e-16 | True |
| 3 | mistral_fewshot | chatgpt_fewshot | 795 | 61 | 187 | -0.1585 | chatgpt_fewshot | 4.648e-16 | 1.069e-14 | True |
| 3 | mistral_fewshot | chatgpt_zeroshot | 795 | 57 | 190 | -0.1673 | chatgpt_zeroshot | 6.702e-18 | 1.608e-16 | True |
| 3 | mistral_fewshot | gemini_fewshot | 795 | 76 | 196 | -0.1509 | gemini_fewshot | 2.185e-13 | 4.807e-12 | True |
| 3 | mistral_fewshot | gemini_zeroshot | 795 | 87 | 202 | -0.1447 | gemini_zeroshot | 1.075e-11 | 2.257e-10 | True |
| 3 | mistral_lora | chatgpt_fewshot | 795 | 72 | 83 | -0.0138 | chatgpt_fewshot | 0.4219 | 1.0000 | False |
| 3 | mistral_lora | chatgpt_zeroshot | 795 | 64 | 82 | -0.0226 | chatgpt_zeroshot | 0.1592 | 1.0000 | False |
| 3 | mistral_lora | gemini_fewshot | 795 | 79 | 84 | -0.0063 | gemini_fewshot | 0.7542 | 1.0000 | False |
| 3 | mistral_lora | gemini_zeroshot | 795 | 92 | 92 | 0.0000 |  | 1.0000 | 1.0000 | False |
| 3 | chatgpt_fewshot | chatgpt_zeroshot | 795 | 23 | 30 | -0.0088 | chatgpt_zeroshot | 0.4101 | 1.0000 | False |
| 3 | chatgpt_fewshot | gemini_fewshot | 795 | 57 | 51 | 0.0075 | chatgpt_fewshot | 0.6306 | 1.0000 | False |
| 3 | chatgpt_fewshot | gemini_zeroshot | 795 | 70 | 59 | 0.0138 | chatgpt_fewshot | 0.3787 | 1.0000 | False |
| 3 | chatgpt_zeroshot | gemini_fewshot | 795 | 62 | 49 | 0.0164 | chatgpt_zeroshot | 0.2546 | 1.0000 | False |
| 3 | chatgpt_zeroshot | gemini_zeroshot | 795 | 71 | 53 | 0.0226 | chatgpt_zeroshot | 0.1265 | 1.0000 | False |
| 3 | gemini_fewshot | gemini_zeroshot | 795 | 44 | 39 | 0.0063 | gemini_fewshot | 0.6609 | 1.0000 | False |
