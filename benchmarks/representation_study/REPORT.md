# TwinAIGym Representation and Causality Study

- Latent tasks: 120
- Trials per task: 3
- Training tasks for Q-learning: 300
- Total evaluated episodes: 12960

## Results

| Representation | Observation | Agent | Success | Reward | Consistency | Recovery | Attribution F1 | Steps | Invalid | Runtime ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_causal | full | heuristic | 100.0% | 3.691 +/- 0.172 | 100.0% | 100.0% | 0.868 | 13.62 | 0.00% | 0.645 |
| graph_causal | full | q_learning | 12.5% | 1.108 +/- 0.190 | 100.0% | 12.5% | 0.477 | 17.38 | 0.00% | 1.066 |
| graph_causal | full | random | 1.4% | -1.164 +/- 0.158 | 95.8% | 1.4% | 0.480 | 18.67 | 18.60% | 0.862 |
| graph_causal | noisy | heuristic | 100.0% | 3.694 +/- 0.172 | 100.0% | 100.0% | 0.864 | 13.53 | 0.00% | 0.928 |
| graph_causal | noisy | q_learning | 80.0% | 3.365 +/- 0.175 | 65.8% | 80.0% | 0.817 | 13.33 | 0.00% | 0.988 |
| graph_causal | noisy | random | 2.5% | -1.169 +/- 0.159 | 94.2% | 2.5% | 0.486 | 18.59 | 18.62% | 1.218 |
| graph_causal | partial | heuristic | 100.0% | 3.739 +/- 0.169 | 100.0% | 100.0% | 0.868 | 12.03 | 0.00% | 0.586 |
| graph_causal | partial | q_learning | 20.8% | 1.703 +/- 0.152 | 100.0% | 20.8% | 0.719 | 16.47 | 0.00% | 0.980 |
| graph_causal | partial | random | 5.8% | -1.120 +/- 0.176 | 88.3% | 5.8% | 0.596 | 18.31 | 22.45% | 1.000 |
| graph_no_propagation | full | heuristic | 100.0% | 3.738 +/- 0.172 | 100.0% | 100.0% | 0.849 | 12.12 | 0.00% | 0.575 |
| graph_no_propagation | full | q_learning | 42.5% | 2.750 +/- 0.187 | 100.0% | 42.5% | 0.660 | 13.57 | 0.00% | 0.779 |
| graph_no_propagation | full | random | 9.4% | 0.288 +/- 0.176 | 80.0% | 9.4% | 0.521 | 18.21 | 18.17% | 0.810 |
| graph_no_propagation | noisy | heuristic | 100.0% | 3.742 +/- 0.172 | 100.0% | 100.0% | 0.849 | 12.00 | 0.00% | 0.782 |
| graph_no_propagation | noisy | q_learning | 90.0% | 3.649 +/- 0.168 | 80.8% | 90.0% | 0.835 | 10.73 | 0.00% | 0.734 |
| graph_no_propagation | noisy | random | 7.2% | 0.241 +/- 0.183 | 84.2% | 7.2% | 0.524 | 18.28 | 18.38% | 1.108 |
| graph_no_propagation | partial | heuristic | 100.0% | 3.781 +/- 0.169 | 100.0% | 100.0% | 0.849 | 10.70 | 0.00% | 0.557 |
| graph_no_propagation | partial | q_learning | 30.0% | 2.760 +/- 0.156 | 100.0% | 30.0% | 0.775 | 15.10 | 0.00% | 0.933 |
| graph_no_propagation | partial | random | 12.5% | 0.221 +/- 0.196 | 78.3% | 12.5% | 0.660 | 17.88 | 22.16% | 0.915 |
| object | full | heuristic | 100.0% | 3.691 +/- 0.172 | 100.0% | 100.0% | 0.868 | 13.62 | 0.00% | 1.503 |
| object | full | q_learning | 12.5% | 1.108 +/- 0.190 | 100.0% | 12.5% | 0.477 | 17.38 | 0.00% | 2.571 |
| object | full | random | 1.4% | -1.164 +/- 0.158 | 95.8% | 1.4% | 0.480 | 18.67 | 18.60% | 1.905 |
| object | noisy | heuristic | 100.0% | 3.693 +/- 0.172 | 100.0% | 100.0% | 0.864 | 13.56 | 0.00% | 1.811 |
| object | noisy | q_learning | 75.3% | 3.297 +/- 0.172 | 66.7% | 75.3% | 0.817 | 13.79 | 0.00% | 1.938 |
| object | noisy | random | 2.8% | -1.211 +/- 0.162 | 91.7% | 2.8% | 0.486 | 18.64 | 18.85% | 1.990 |
| object | partial | heuristic | 100.0% | 3.739 +/- 0.169 | 100.0% | 100.0% | 0.868 | 12.03 | 0.00% | 0.941 |
| object | partial | q_learning | 20.8% | 1.703 +/- 0.152 | 100.0% | 20.8% | 0.719 | 16.47 | 0.00% | 1.448 |
| object | partial | random | 5.8% | -1.120 +/- 0.176 | 88.3% | 5.8% | 0.596 | 18.31 | 22.45% | 1.250 |
| tabular | full | heuristic | 100.0% | 3.691 +/- 0.172 | 100.0% | 100.0% | 0.868 | 13.62 | 0.00% | 0.707 |
| tabular | full | q_learning | 12.5% | 1.108 +/- 0.190 | 100.0% | 12.5% | 0.477 | 17.38 | 0.00% | 1.055 |
| tabular | full | random | 1.4% | -1.164 +/- 0.158 | 95.8% | 1.4% | 0.480 | 18.67 | 18.60% | 0.862 |
| tabular | noisy | heuristic | 100.0% | 3.694 +/- 0.172 | 100.0% | 100.0% | 0.864 | 13.53 | 0.00% | 0.919 |
| tabular | noisy | q_learning | 80.0% | 3.365 +/- 0.175 | 65.8% | 80.0% | 0.817 | 13.33 | 0.00% | 0.947 |
| tabular | noisy | random | 2.5% | -1.169 +/- 0.159 | 94.2% | 2.5% | 0.486 | 18.59 | 18.62% | 1.220 |
| tabular | partial | heuristic | 100.0% | 3.739 +/- 0.169 | 100.0% | 100.0% | 0.868 | 12.03 | 0.00% | 0.729 |
| tabular | partial | q_learning | 20.8% | 1.703 +/- 0.152 | 100.0% | 20.8% | 0.719 | 16.47 | 0.00% | 1.304 |
| tabular | partial | random | 5.8% | -1.120 +/- 0.176 | 88.3% | 5.8% | 0.596 | 18.31 | 22.45% | 1.149 |

## Interpretation guardrails

- Every condition is generated from the same latent tasks and action vocabulary.
- `graph_no_propagation` is an ablation of dynamics, not merely serialization.
- Tabular and object conditions retain causal dynamics; they isolate representation.
- The forced first-step symptom restart measures recovery after a controlled error.
- LLM conditions are omitted when no API key is configured; they must not be imputed.

## Scalability

| Representation | Nodes | Relations | Observation ms | Transition ms | Payload bytes |
|---|---:|---:|---:|---:|---:|
| graph_causal | 10 | 11 | 0.028 | 0.020 | 3138 |
| graph_no_propagation | 10 | 11 | 0.028 | 0.016 | 3146 |
| tabular | 10 | 11 | 0.028 | 0.018 | 3063 |
| object | 10 | 11 | 0.039 | 0.018 | 4241 |
| graph_causal | 25 | 38 | 0.083 | 0.060 | 8467 |
| graph_no_propagation | 25 | 38 | 0.136 | 0.051 | 8475 |
| tabular | 25 | 38 | 0.079 | 0.055 | 8058 |
| object | 25 | 38 | 0.143 | 0.043 | 12432 |
| graph_causal | 50 | 78 | 0.200 | 0.118 | 17142 |
| graph_no_propagation | 50 | 78 | 0.193 | 0.097 | 17150 |
| tabular | 50 | 78 | 0.229 | 0.141 | 16236 |
| object | 50 | 78 | 0.418 | 0.110 | 25082 |
| graph_causal | 100 | 157 | 0.632 | 0.332 | 34249 |
| graph_no_propagation | 100 | 157 | 0.568 | 0.296 | 34257 |
| tabular | 100 | 157 | 0.574 | 0.316 | 32365 |
| object | 100 | 157 | 1.371 | 0.349 | 50509 |

## Paired contrasts

Positive deltas favor the reference representation.

| Reference | Comparison | Observation | Agent | Pairs | Success delta | Reward delta | 95% CI |
|---|---|---|---|---:|---:|---:|---:|
| graph_causal | tabular | full | heuristic | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | full | q_learning | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | full | random | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | noisy | heuristic | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | noisy | q_learning | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | noisy | random | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | partial | heuristic | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | partial | q_learning | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | tabular | partial | random | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | object | full | heuristic | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | object | full | q_learning | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | object | full | random | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | object | noisy | heuristic | 360 | +0.0% | +0.001 | +/- 0.002 |
| graph_causal | object | noisy | q_learning | 360 | +4.7% | +0.068 | +/- 0.118 |
| graph_causal | object | noisy | random | 360 | -0.3% | +0.042 | +/- 0.133 |
| graph_causal | object | partial | heuristic | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | object | partial | q_learning | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | object | partial | random | 360 | +0.0% | +0.000 | +/- 0.000 |
| graph_causal | graph_no_propagation | full | heuristic | 360 | +0.0% | -0.047 | +/- 0.006 |
| graph_causal | graph_no_propagation | full | q_learning | 360 | -30.0% | -1.642 | +/- 0.194 |
| graph_causal | graph_no_propagation | full | random | 360 | -8.1% | -1.452 | +/- 0.102 |
| graph_causal | graph_no_propagation | noisy | heuristic | 360 | +0.0% | -0.048 | +/- 0.005 |
| graph_causal | graph_no_propagation | noisy | q_learning | 360 | -10.0% | -0.284 | +/- 0.102 |
| graph_causal | graph_no_propagation | noisy | random | 360 | -4.7% | -1.410 | +/- 0.102 |
| graph_causal | graph_no_propagation | partial | heuristic | 360 | +0.0% | -0.042 | +/- 0.004 |
| graph_causal | graph_no_propagation | partial | q_learning | 360 | -9.2% | -1.057 | +/- 0.199 |
| graph_causal | graph_no_propagation | partial | random | 360 | -6.7% | -1.341 | +/- 0.101 |

## Skipped conditions

- llm_direct/graph_causal/full: missing API key
- llm_causal/graph_causal/full: missing API key
- llm_audit/graph_causal/full: missing API key
- llm_direct/graph_causal/partial: missing API key
- llm_causal/graph_causal/partial: missing API key
- llm_audit/graph_causal/partial: missing API key
- llm_direct/graph_causal/noisy: missing API key
- llm_causal/graph_causal/noisy: missing API key
- llm_audit/graph_causal/noisy: missing API key
- llm_direct/graph_no_propagation/full: missing API key
- llm_causal/graph_no_propagation/full: missing API key
- llm_audit/graph_no_propagation/full: missing API key
- llm_direct/graph_no_propagation/partial: missing API key
- llm_causal/graph_no_propagation/partial: missing API key
- llm_audit/graph_no_propagation/partial: missing API key
- llm_direct/graph_no_propagation/noisy: missing API key
- llm_causal/graph_no_propagation/noisy: missing API key
- llm_audit/graph_no_propagation/noisy: missing API key
- llm_direct/tabular/full: missing API key
- llm_causal/tabular/full: missing API key
- llm_audit/tabular/full: missing API key
- llm_direct/tabular/partial: missing API key
- llm_causal/tabular/partial: missing API key
- llm_audit/tabular/partial: missing API key
- llm_direct/tabular/noisy: missing API key
- llm_causal/tabular/noisy: missing API key
- llm_audit/tabular/noisy: missing API key
- llm_direct/object/full: missing API key
- llm_causal/object/full: missing API key
- llm_audit/object/full: missing API key
- llm_direct/object/partial: missing API key
- llm_causal/object/partial: missing API key
- llm_audit/object/partial: missing API key
- llm_direct/object/noisy: missing API key
- llm_causal/object/noisy: missing API key
- llm_audit/object/noisy: missing API key
