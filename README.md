# 🌹 Golestan Clustering Analysis

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Persian-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview

This project applies **Unsupervised Machine Learning** techniques to analyze and cluster the text of **Saadi's Golestan**, one of the most significant literary works in the Persian language.

The goal is to group anecdotes (*Hikayats*) based on semantic similarity and compare the machine-generated clusters with the original 8 chapters (*Babs*) defined by Saadi. We utilize **TF-IDF** for vectorization, **LSA** for dimensionality reduction, and compare **K-Means** vs. **DBSCAN** algorithms.

## 🚀 Key Features

*   **Advanced Persian Preprocessing**:
    *   Normalization and Tokenization using `Hazm`.
    *   Removal of general stopwords and a curated list of **archaic/literary stopwords** specific to classical Persian texts (e.g., "گفت", "ملک", "شنیدم").
    *   Lemmatization to reduce word variations.
*   **Feature Extraction**: TF-IDF Vectorization with L2 Normalization.
*   **Dimensionality Reduction**: Latent Semantic Analysis (LSA/TruncatedSVD) to handle sparsity and improve clustering performance.
*   **Clustering Algorithms**:
    *   **K-Means**: With Elbow Method and Silhouette Analysis to find optimal $k$.
    *   **DBSCAN**: Density-based clustering with automatic Epsilon determination using K-distance graphs.
*   **Visualization**: Confusion Matrices (Heatmaps) to evaluate cluster alignment with real chapters.

## 📂 Project Structure

```text
Golestan_Clustering/
│
├── data/
│   └── golestan.csv       # The dataset containing anecdotes and labels
│
├── outputs/               # Saved visualizations (optional)
│
├── main.ipynb             # The main Jupyter Notebook with all logic
│
├── README.md              # Project documentation
│
└── requirements.txt       # Python dependencies‍‍‍‍‍‍‍
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mdvr0480/Golestan-Clustering.git
    cd Golestan-Clustering
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Linux/Mac
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Methodology & Results

### 1. Preprocessing
We cleaned the text by removing punctuation and high-frequency verbs that do not carry semantic meaning in the context of topic modeling. This ensures the model focuses on themes (e.g., "Justice", "Love", "Education") rather than grammar.

### 2. K-Means Analysis
*   **Optimal K:** Using the Elbow Method and Silhouette Score, we identified the optimal number of clusters (around $k=8$).
*   **Interpretation:** The resulting clusters showed overlap with the original chapters, successfully grouping stories about "Kings and Rulers" separately from stories about "Dervishes" or "Education".

### 3. DBSCAN Analysis
*   **Parameter Tuning:** We used a **K-distance graph** to find the optimal `eps` value.
*   **Noise Handling:** DBSCAN identified outliers (Label -1), which is expected in literary texts where some anecdotes are unique and do not fit strictly into dense semantic clusters.

## 📈 Visualizations

The project generates several key plots:
*   **Elbow & Silhouette Graph:** To decide the number of clusters.
*   **K-distance Graph:** To tune DBSCAN.
*   **Confusion Matrix Heatmap:** To visualize the correlation between predicted clusters and actual chapters.

## 📦 Libraries Used

*   **Pandas & NumPy**: Data manipulation.
*   **Scikit-Learn**: Machine Learning algorithms (KMeans, DBSCAN, TF-IDF, SVD).
*   **Hazm**: Persian NLP toolkit.
*   **Matplotlib & Seaborn**: Data visualization.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.