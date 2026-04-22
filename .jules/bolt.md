## 2024-05-14 - [O(N^2) Bottleneck in Sequential Embeddings]
**Learning:** `FormulaEmbedder._encode_tfidf` was unconditionally refitting the TF-IDF vectorizer and re-encoding the entire embeddings cache on every new factor added. In a sequential processing loop, this created a severe O(N^2) bottleneck.
**Action:** Before refitting a vectorizer in a sequential loop, use `build_analyzer()` to extract query tokens and check against `vocabulary_`. If the vocabulary hasn't grown, you can safely skip the refit and just `transform([text])` the new item, turning O(N^2) behavior back to O(N).
