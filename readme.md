Loading Russian RAG dataset...<br>
Evaluating on 923 samples from Russian RAG dataset...<br>
Evaluating Russian RAG dataset: 100%|██████████| 923/923 [41:24<00:00,  2.69s/it]

Evaluation complete! Results saved to evaluation_results.csv

=== Evaluation Summary ===<br>
Total questions evaluated: 923<br>
Correct answers: 330<br>
Accuracy: 0.36<br>

=== Example Results ===

Question: На каких платформах выйдет Diablo 4?<br>
Correct answer: PC, PlayStation 4, Xbox One<br>
Generated answer: PC, PlayStation 4, PlayStation 5, Xbox One, Xbox Series X/S.<br>
Correct: False<br>

Question: Какой по счёту планетой от Солнца является Нептун?<br>
Correct answer: 8<br>
Generated answer: Восьмой.<br>
Correct: False<br>

Question: Какая болезнь развилась у Бетховена в последние годы жизни?<br>
Correct answer: Глухота<br>
Generated answer: I don't know.<br>
Correct: False<br>

***

Loading and sampling 100000 examples from datasets/v1.0-simplified_nq-dev-all.jsonl/v1.0-simplified_nq-dev-all.jsonl...<br>
Warning: JSON decode error on line 1312: Unterminated string starting at: line 1 column 1346 (char 1345)<br>
Encountered 1 errors while loading the dataset<br>
Successfully loaded 1311 valid examples<br>
Evaluating on 1311 examples...<br>
Evaluating NQ: 100%|██████████| 1311/1311 [33:43<00:00,  1.54s/it]<br>

Evaluation complete. Accuracy: 33.01%<br>
Results saved to evaluation_results_simplified.csv
