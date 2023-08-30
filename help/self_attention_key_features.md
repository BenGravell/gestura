1. **Main Idea**: Instead of focusing only on a fixed context or window of tokens (like in many RNNs), self-attention allows the model to focus on all tokens of the input sequence, assigning different attention scores to each token.

2. **Attention Score**: For each token, a set of queries, keys, and values are computed. The attention score between two tokens is computed by taking the dot product of their query and key, followed by a scaling operation and softmax.

3. **Weighted Sum**: The output for each token is then a weighted sum of all values, where the weights are the computed attention scores.

4. **Multi-head Attention**: Instead of having one set of attention weights, the model can have multiple sets (heads). Each set produces different attention patterns, and their outputs are concatenated and linearly transformed.

5. **Benefits**:
   - Captures long-range dependencies in data.
   - Enables parallel processing of sequences, unlike RNNs.
   - Has led to state-of-the-art results in various NLP tasks.
