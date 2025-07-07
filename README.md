# iDRNMF
## Instance-wise Distributionally Robust Nonnegative Matrix Factorization

  *Wafa Barkhoda, Amjad Seyedi, Nicolas Gillis, and Fardin Akhlaghian Tab*
  
  *Pattern Recognition 2026*
  
  *https://doi.org/10.1016/j.patcog.2025.111732*

  # Abstract

Nonnegative matrix factorization (NMF) stands as a prevalent algebraic representation technique deployed across diverse domains such as data mining and machine learning. At its core, NMF aims to minimize the distance between the original input and a lower-rank approximation of it. However, when data is noisy or contains outliers, NMF struggles to provide accurate results. Existing robust methods rely on known distribution assumptions, which limit their effectiveness in real-world situations where the noise distribution is unknown. To address this gap, we introduce a new model, called instance-wise distributionally robust NMF (iDRNMF), that can handle a wide range of noise distributions. By leveraging a weighted sum multi-objective method, iDRNMF can handle multiple noise distributions and their combinations. Furthermore, while the entry-wise models assume noise contamination at the individual matrix entries level, the proposed instance-wise model assumes noise contamination at the entire data instances level (columns of the input matrix). This instance-wise model is often more appropriate for data representation tasks, as it addresses the noise affecting entire feature vectors rather than individual features. To train this model, we develop a unified multi-objective optimization framework based on an iterative reweighted algorithm, which maintains computational efficiency similar to single-objective NMFs. This framework provides flexible updating rules, making it suitable for optimizing a wide range of robust and distributionally robust objectives. Extensive experiments on various datasets with distinct noise distributions and mixtures thereof show the superior performance of iDRNMF compared to state-of-the-art models, showcasing its effectiveness in handling diverse noise profiles on real-world problems.
