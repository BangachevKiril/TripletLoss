### Triplet Loss




References:
- Original Paper: https://arxiv.org/pdf/1503.03832



- Extension to ladder loss: https://qilin-zhang.github.io/_pages/pdfs/Ladder_Loss_for_Coherent_Visual-Semantic_Embedding.pdf.
Hierarchical based on relevance degree. I.e., some classes might be more relevant than others. So we have C_0,C_1,C_2, \ldots, C_k,
each subsequent is less relevant w.r.t C_0. Now, we have for a \in C_0:
 \sum_{p\in C_0, n\in C_{\ge 1}} [d(a,p) -d(a,n)+m_1]_+ 
+\sum_{p\in C_1, n\in C_{\ge 2}} [d(a,p) -d(a,n)+m_2]_+ 
+\cdots
+\sum_{p\in C_{k-1}, n\in C_{k}} [d(a,p) -d(a,n)+m_{k-1}]_+ 
