# CLAIMFAST
## Experimental Results and Performance Analysis

**Report Date:** 2026-11-15
**Project:** Insurance Claim Classification using LLMs

---

# 1. Introduction

ClaimFast is an experimental project designed to explore the use of Large Language Models (LLMs) for classifying insurance claim messages and extracting structured information.

The experiment compares five prompting strategies: Zero-Shot, Few-Shot with 3 examples, Few-Shot with 5 examples, Few-Shot with 8 examples, and Chain-of-Thought. The objective is to evaluate their classification performance, information extraction quality, safety-related detection, processing latency, and token usage.

# 2. Experimental Methodology

The experiment used a synthetic and anonymised dataset containing 80 labelled insurance-related messages. The dataset was divided into:

- **Development dataset:** 20 records, used for prompt development and selection of few-shot examples.
- **Held-out test dataset:** 60 records, reserved for final evaluation.

Five prompting strategies were evaluated:

- **Zero-Shot:** The model received task instructions without labelled examples.
- **Few-Shot 3:** The prompt included 3 labelled examples selected from the development dataset.
- **Few-Shot 5:** The prompt included 5 labelled examples selected from the development dataset.
- **Few-Shot 8:** The prompt included 8 labelled examples selected from the development dataset.
- **Chain-of-Thought:** The prompt used a reasoning-oriented instruction followed by a request for the final structured JSON response.

All five strategies used the same model, llama3.2:3b, through the local Ollama API. The temperature was set to 0 to reduce variation in model responses.

Each strategy was evaluated three times using the same 60 held-out test records. This resulted in 15 experiment runs and 900 total inference requests (5 strategies × 3 runs × 60 records).

Raw model responses, parsed predictions, JSON validity, parsing errors, latency, and token usage were recorded in JSONL output files. The held-out test dataset was kept separate from prompt development and was used for final evaluation.

# 3. Evaluation Metrics

The following metrics were used to compare the prompting strategies:

- **JSON Validity:** Percentage of model responses that produced valid JSON output.
- **Classification Accuracy:** Percentage of messages assigned the correct claim category.
- **Macro F1-Score:** Average F1-score across claim categories, giving equal importance to each category.
- **Confusion Matrix:** Shows correct classifications and the types of classification errors across claim categories.
- **Extraction Exact Match:** Percentage of records for which all evaluated extracted fields exactly matched the labelled values.
- **Field-Level Precision, Recall and F1-Score:** Measures extraction performance separately for individual fields.
- **Hallucination Rate:** Measures unsupported extracted values, such as dates, locations, or other information not supported by the message.
- **Injury Recall:** Measures the proportion of labelled injury cases correctly detected.
- **High-Urgency Recall:** Measures the proportion of labelled high-urgency cases correctly identified.
- **Run-to-Run Agreement:** Measures consistency of predictions across the three repeated runs.
- **Average Latency:** Average processing time in seconds per message.
- **Average Total Tokens:** Average combined input and output token usage per message.

Overall classification and extraction comparison metrics were calculated across three runs. Field-level extraction metrics and injury/high-urgency recall were calculated using Run 1. This distinction is maintained when interpreting the results in the following sections.

# 4. Final Held-Out Test Results

The following table presents the final evaluation results on the 60-record held-out test dataset. Each prompting strategy was evaluated three times using the same test records.

| Experiment | Accuracy | Macro F1 | Extraction Exact Match |
|---|---:|---:|---:|
| Zero-Shot | 93.3% | 0.940 | 48.3% |
| Few-Shot 3 | 83.3% | 0.836 | 58.3% |
| Few-Shot 5 | 91.7% | 0.933 | 63.3% |
| Few-Shot 8 | 95.0% | 0.954 | 68.3% |
| Chain-of-Thought | 96.7% | 0.970 | 51.7% |

The results show that Chain-of-Thought achieved the highest classification accuracy and Macro F1-score. Few-Shot 8 achieved the highest extraction exact-match score.

These findings indicate that classification performance and information extraction performance are different aspects of the task. A strategy that performs well in classification does not necessarily produce the best structured extraction results.

# 5. Classification Accuracy Analysis

The final classification accuracy was calculated using 60 held-out test records for each strategy.

| Strategy | Correct Predictions | Accuracy |
|---|---:|---:|
| Zero-Shot | 56/60 | 93.3% |
| Few-Shot 3 | 50/60 | 83.3% |
| Few-Shot 5 | 55/60 | 91.7% |
| Few-Shot 8 | 57/60 | 95.0% |
| Chain-of-Thought | 58/60 | 96.7% |

Chain-of-Thought achieved the highest classification accuracy at 96.7%, followed by Few-Shot 8 at 95.0%.

Few-Shot 3 performed below Zero-Shot, while Few-Shot 5 also remained slightly below Zero-Shot. This indicates that increasing the number of examples does not automatically improve classification performance.

# 6. Macro F1-Score Analysis

| Strategy | Macro F1-Score |
|---|---:|
| Zero-Shot | 0.940 |
| Few-Shot 3 | 0.836 |
| Few-Shot 5 | 0.933 |
| Few-Shot 8 | 0.954 |
| Chain-of-Thought | 0.970 |

Chain-of-Thought achieved the highest Macro F1-score of 0.970, indicating strong classification performance across the claim categories.

Few-Shot 8 followed with a score of 0.954. Few-Shot 3 recorded the lowest Macro F1-score at 0.836.

# 7. Safety and High-Risk Detection Analysis

Safety-related performance was evaluated using injury detection recall and high-urgency recall on the held-out test dataset. These metrics were calculated using Run 1 of each strategy.

| Strategy | Injury Recall | High-Urgency Recall |
|---|---:|---:|
| Zero-Shot | 83.33% | 100.00% |
| Few-Shot 3 | 100.00% | 75.00% |
| Few-Shot 5 | 83.33% | 62.50% |
| Few-Shot 8 | 83.33% | 62.50% |
| Chain-of-Thought | 100.00% | 87.50% |

Zero-Shot detected all 8 high-urgency cases, achieving 100% high-urgency recall. However, it detected only 5 of the 6 injury cases.

Few-Shot 3 and Chain-of-Thought detected all 6 injury cases. Chain-of-Thought also achieved 87.50% high-urgency recall by detecting 7 of the 8 high-urgency cases.

Few-Shot 5 and Few-Shot 8 achieved 62.50% high-urgency recall, indicating that they missed 3 out of 8 high-urgency cases.

These results demonstrate that classification accuracy and extraction performance alone are insufficient for evaluating an insurance claim-processing system. Missing an injury-related or high-urgency case could have serious operational consequences.

For production deployment, ClaimFast should use risk-based routing, with human review for injury-related, high-urgency, ambiguous, or low-confidence messages. Safety thresholds should be validated using a larger, independently labelled dataset.

*Note: These safety metrics are based on Run 1 and should not be interpreted as three-run aggregate metrics.*

# 8. Processing Latency Analysis

The average processing latency was calculated across three runs of the 60-record held-out test dataset.

| Strategy | Average Latency (seconds/message) |
|---|---:|
| Zero-Shot | 1.517 |
| Few-Shot 3 | 2.431 |
| Few-Shot 5 | 2.089 |
| Few-Shot 8 | 3.088 |
| Chain-of-Thought | 3.020 |

Zero-Shot recorded the lowest average latency at approximately 1.517 seconds per message. Few-Shot 8 recorded the highest average latency at approximately 3.088 seconds per message.

These results indicate that larger prompting configurations can require more processing time. However, latency should be considered alongside classification accuracy and extraction quality.

# 9. Token Usage Analysis

The following table presents the average total token usage per message across three runs.

| Strategy | Average Total Tokens per Message |
|---|---:|
| Zero-Shot | 455.37 |
| Few-Shot 3 | 607.73 |
| Few-Shot 5 | 659.50 |
| Few-Shot 8 | 1475.50 |
| Chain-of-Thought | 418.47 |

Few-Shot 8 consumed the highest average number of tokens, at approximately 1475.50 tokens per message. Chain-of-Thought recorded the lowest average total token usage at approximately 418.47 tokens per message in this experiment.

Token usage depends on both the input prompt and the generated response. These figures represent average total tokens, not separate prompt and completion token counts. Across all 900 inference requests, the experiment used approximately 650,982 combined input and output tokens, averaging about 723 tokens per request.

# 10. Field-Level Accuracy Analysis

The following table presents the exact-match rate for each extracted field. These field-level metrics were calculated using Run 1 of each strategy.

| Field | Zero-Shot | Few-Shot 3 | Few-Shot 5 | Few-Shot 8 | Chain-of-Thought |
|---|---:|---:|---:|---:|---:|
| estimated_amount_inr | 100.00% | 98.33% | 100.00% | 98.33% | 98.33% |
| incident_date | 96.67% | 96.67% | 95.00% | 86.67% | 95.00% |
| injury_reported | 96.67% | 95.00% | 96.67% | 96.67% | 98.33% |
| location | 95.00% | 93.33% | 93.33% | 93.33% | 95.00% |
| police_report_filed | 95.00% | 96.67% | 96.67% | 96.67% | 95.00% |
| policy_number | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| urgency | 63.33% | 71.67% | 78.33% | 91.67% | 65.00% |
| vehicle_reg_no | 91.67% | 96.67% | 91.67% | 96.67% | 91.67% |

The results show that policy number extraction achieved 100% exact-match accuracy across all five strategies.

Few-Shot 8 achieved the highest urgency exact-match rate at 91.67%, while Few-Shot 3 and Few-Shot 8 achieved the highest vehicle registration exact-match rate at 96.67%.

Few-Shot 8 also had the lowest incident-date exact-match rate at 86.67%. Its evaluation recorded eight incident-date hallucinations, showing that a strategy can perform well on one field while making more errors on another.

Chain-of-Thought achieved 98.33% exact match for injury information and 95.00% for location extraction.

These results demonstrate that extraction performance varies significantly between fields and prompting strategies. Field-level precision, recall, F1-score, and hallucination statistics are also available in the generated evaluation file at outputs/evaluation/per_field_metrics.csv.

*Note: Field-level metrics are based on Run 1, whereas the overall strategy comparison uses averages across three runs. Claim classification is evaluated separately in the classification results.*

## 10.1 Output Reliability and Run-to-Run Agreement

All five strategies produced valid JSON in all three runs. Each run contained 60 records, giving 180 valid JSON responses per strategy.

| Strategy | Valid JSON Responses | JSON Validity | Valid Classifications per Run | Run Agreement |
|---|---:|---:|---:|---:|
| Zero-Shot | 180/180 | 100% | 60/60 | 100% |
| Few-Shot 3 | 180/180 | 100% | 59/60 | 100% |
| Few-Shot 5 | 180/180 | 100% | 58/60 | 100% |
| Few-Shot 8 | 180/180 | 100% | 60/60 | 100% |
| Chain-of-Thought | 180/180 | 100% | 60/60 | 100% |

Run-to-run agreement was 100% for all strategies: the predictions for all 60 records were identical across the three runs.

Valid JSON does not necessarily mean that the response contains a valid claim classification. Few-Shot 3 and Few-Shot 5 produced valid JSON but had one and two invalid or missing claim classifications respectively in each run.

The perfect run agreement indicates repeatability under the experimental setup. It does not establish that every repeated prediction was correct.

## 10.2 Confusion Matrix Analysis

Confusion matrices were generated for each strategy using Run 1 of the held-out test dataset. The matrices compare the labelled claim category with the model's predicted category.

| Strategy | Correct Classifications | Classification Errors |
|---|---:|---:|
| Zero-Shot | 56 | 4 |
| Few-Shot 3 | 50 | 10 |
| Few-Shot 5 | 55 | 5 |
| Few-Shot 8 | 57 | 3 |
| Chain-of-Thought | 58 | 2 |

The Few-Shot 3 matrix showed notable confusion between third-party liability and own-damage claims, with seven third-party liability records predicted as own damage.

Zero-Shot confused two own-damage records with glass-only claims. Few-Shot 8 had one own-damage record predicted as glass-only and one theft record predicted as glass-only. Chain-of-Thought had two errors: one third-party liability record and one glass-only record were predicted as own damage.

The complete confusion matrices for all three runs are available as CSV files in outputs/evaluation/. The summary above reports Run 1 and should not be interpreted as an average confusion matrix across all runs.

## 10.3 Error Analysis

## 10.3 Error Analysis

The error analysis examined 132 error records across the five prompting strategies. From these, 75 records were selected for detailed analysis, with 15 records per strategy from Run 1.

The selected records were checked by comparing their gold JSON values with the corresponding predicted JSON values. This consistency check identified 26 records where the originally listed mismatch fields did not fully agree with the actual JSON differences. The mismatch-field lists and error categories were updated accordingly.

The corrected set of 75 records contained the following error categories:

| Error Category | Number of Records |
|---|---:|
| Extraction error | 55 |
| Classification error | 10 |
| Classification and extraction error | 10 |
| **Total** | **75** |

A total of 94 individual root-cause assignments were identified because some records contained multiple root causes. Their distribution was:

| Root Cause | Frequency |
|---|---:|
| Urgency rule application error | 34 |
| Claim-type boundary confusion | 20 |
| Location extraction or over-extraction error | 10 |
| Vehicle-registration extraction or formatting error | 10 |
| Police-report status extraction error | 8 |
| Date extraction or normalization error | 7 |
| Injury-status extraction error | 5 |
| **Total root-cause assignments** | **94** |

Incorrect urgency was the most frequently identified issue. Claim-type boundary confusion was the second most frequent. These findings indicate that applying consistent urgency criteria and distinguishing between similar claim categories were important challenges.

Other observed issues included extracting generic settings as specific incident locations, confusing policy numbers with vehicle registration numbers, omitting explicit negative injury information, incorrectly handling police-report status, and generating unsupported incident dates.

The proposed improvements include clearer urgency definitions, contrastive examples for similar claim types, stricter vehicle registration validation, and explicit instructions to avoid unsupported date, location, and police-report extraction.

The corrected error-analysis sheet contains the gold and predicted JSON values, verified mismatch fields, suggested root causes, severity assessments, and proposed fixes for all 75 selected records.

This analysis uses an AI-assisted consistency check of the gold and predicted JSON values. The root-cause explanations and proposed fixes are suggested interpretations and have not been independently validated by a second human reviewer. Therefore, this work should not be considered a completed independent manual review.

# 11. Experimental Visualization

The following dashboard is retained from the preliminary development experiments. It represents the earlier four-strategy comparison and should not be interpreted as the final five-strategy held-out test evaluation.

![ClaimFast Preliminary Experiment Dashboard](outputs/experiment_comparison.png)

The final held-out test results are reported in the tables in Sections 4–10 and in the evaluation files under outputs/evaluation/.

# 12. Discussion

The final held-out test results show that prompting strategy affects classification accuracy, information extraction, processing latency, and token usage.

Chain-of-Thought achieved the highest classification accuracy of 96.67% and the highest Macro F1-score of 0.970. Few-Shot 8 followed with 95.00% accuracy and a Macro F1-score of 0.954. However, Chain-of-Thought did not achieve the highest extraction exact-match rate.

Few-Shot 8 achieved the highest extraction exact-match rate at 68.33%. It also achieved the highest urgency field exact-match rate at 91.67%. However, its incident-date exact-match rate was the lowest among the strategies at 86.67%, and eight incident-date hallucinations were recorded in Run 1.

The results for k=3, k=5, and k=8 did not show a consistent improvement at every step. Classification accuracy increased from 83.33% at k=3 to 91.67% at k=5 and 95.00% at k=8. This suggests that additional examples helped in this experiment, but the improvement was not guaranteed for every metric.

Zero-Shot was the fastest strategy, with an average latency of 1.517 seconds per message. Chain-of-Thought recorded the lowest average total token usage, while Few-Shot 8 had the highest token usage and latency.

The safety results also require attention. In Run 1, Zero-Shot achieved 100% high-urgency recall, while Chain-of-Thought achieved 87.50%. Few-Shot 3 and Chain-of-Thought detected all six injury cases. These findings show why overall classification accuracy alone should not determine the production strategy.

The error analysis identified issues involving urgency prediction, location over-extraction, vehicle registration formatting, missing negative values, date hallucinations, and claim classification. A mixed incident involving broken glass and theft was also identified as a classification challenge. The current results do not establish that Chain-of-Thought consistently solves ambiguous or multi-incident cases; targeted testing is required.

For a workload of 5,000 messages per day, Chain-of-Thought is a promising candidate when classification performance is the main priority. However, its measured average latency of approximately 3.020 seconds per message would require roughly 4.2 hours of sequential processing for 5,000 messages under the measured conditions. This is an estimate, not a production throughput guarantee. A deployment decision should include load testing, parallel processing tests, human review, and monitoring. A hybrid strategy may also be explored, but it has not been tested in this experiment.

# 13. Limitations

- The held-out test dataset contains only 60 synthetic records, so the results may not represent real-world insurance claim messages.
- The experiments used one local model, llama3.2:3b, through Ollama. Results may differ with other models or execution environments.
- Each strategy was evaluated over three runs. The small dataset limits the strength of conclusions about general performance.
- Field-level extraction metrics and injury/high-urgency recall are based on Run 1 only, while the overall strategy comparison uses three-run averages.
- The error categories were generated automatically and are marked as suggested classifications requiring human verification. They should not be treated as a completed manual review.
- The experiment did not establish calibrated confidence scores or validate a production confidence threshold.
- The latency measurements do not represent a production load test with 5,000 daily messages or concurrent requests.
- Token usage is reported as average total tokens per message. Actual deployment cost will depend on the model, infrastructure, and pricing arrangement.
- Further testing is needed for multilingual messages, ambiguous incidents, multiple incidents in one message, and unusual claim descriptions.

# 14. Conclusion

This experiment compared Zero-Shot, Few-Shot with k=3, k=5 and k=8, and Chain-of-Thought prompting for the ClaimFast insurance claim-processing task.

On the 60-record held-out test dataset, Chain-of-Thought achieved the highest classification accuracy at 96.67% and the highest Macro F1-score at 0.970. Few-Shot 8 achieved the highest extraction exact-match rate at 68.33%. Zero-Shot was the fastest strategy, with an average latency of approximately 1.517 seconds per message.

No single strategy performed best across every metric. The results demonstrate the importance of evaluating classification, extraction, safety-related recall, latency, and token usage together.

Chain-of-Thought is a promising candidate when classification performance is the primary objective, while Few-Shot 8 showed stronger overall extraction exact match. Before production deployment, ClaimFast should be tested on a larger real-world dataset, with human review for high-risk cases, monitoring for hallucinations, and load testing under realistic workloads.
