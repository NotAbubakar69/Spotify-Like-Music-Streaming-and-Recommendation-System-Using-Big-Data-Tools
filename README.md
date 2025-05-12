# Spotify-Like-Music-Streaming-and-Recommendation-System-Using-Big-Data-Tools

## Overview

This repository presents a web-based music recommendation system leveraging a subset of 2,261 audio files sampled from a 105 GB dataset. The project focuses on scalable, memory-efficient, and high-quality music recommendations by combining advanced feature extraction, efficient nearest neighbor search using the tree-based SCANN (Scalable Nearest Neighbors) algorithm, and hyperparameter tuning with grid search. It was developed as a part of the Fundamentals of Big Data Analytics (DS2004) course.

---

## Dependencies

* Librosa
* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* SCANN
* MongoDB

---

## Introduction

Our music recommendation system distinguishes itself through advanced audio feature extraction and efficient similarity matching:

* *Feature Extraction*: Librosa is used to extract MFCC, Mel spectrogram, and Tempogram features to capture musical nuances.
* *Nearest Neighbor Search*: SCANN is used to find sonically similar tracks based on these features.
* *Hyperparameter Tuning*: Grid Search optimizes SCANN settings.
* *Evaluation*: The Silhouette score is used to measure clustering quality and recommendation performance.

---

## Features

* *Audio Feature Extraction*: Extracts MFCC, Mel spectrograms, and Tempograms.
* *Recommendation System*: Employs SCANN for fast, accurate recommendations based on sonic similarity.
* *Hyperparameter Tuning*: Grid search for optimal SCANN configuration.
* *Evaluation*: Silhouette score assesses clustering and effectiveness.

---

## Limitations of Traditional Nearest Neighbor & LSH-based SCANN

### Nearest Neighbor Model

* *Scalability*: Struggles with large datasets.
* *Memory Usage*: High memory consumption due to dataset size.
* *Curse of Dimensionality*: Performance degrades with high-dimensional data.

### LSH-based SCANN

* *Approximation Errors*: May sacrifice accuracy for speed.
* *Parameter Sensitivity*: Requires careful hash parameter tuning.
* *High-Dimensional Challenge*: Less effective in high-dimensional spaces.

---

## Tree-Based SCANN: Our Solution

* *Hierarchical Structure*: Organizes data into partitions for efficient searching.
* *Balanced Partitioning*: Maintains even data distribution.
* *Dimensionality Reduction*: Splits along high-variance dimensions to improve performance.
* *Adaptive Splitting*: Tailors partitions to data characteristics dynamically.

---

## Usage

1. *Data Preprocessing*

   * Run preprocessing.ipynb to extract audio features and save them to preprocessed_audios.csv.

2. *Model Training*

   * Run recommendation_model.ipynb to build the recommendation model using SCANN.

3. *MongoDB Integration*

   * Use mongodbscript.ipynb to store and manage features in MongoDB.

---

## Why Our Approach is Better

* *Efficiency & Accuracy*: Combines nearest neighbor interpretability with SCANN's performance.
* *Scalability & Memory Use*: Tree-based SCANN scales to large datasets while conserving memory.
* *Robustness & Flexibility*: Supports various datasets and tuning strategies for optimal performance.

---

## Contributors

* M. Muttayab (22i-1949)
* Muhammad Abubakar Nadeem (22i-2003)
