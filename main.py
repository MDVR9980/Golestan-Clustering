import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm

# Persian NLP Libraries
from hazm import Normalizer, word_tokenize, Lemmatizer, stopwords_list
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning & Clustering Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.metrics import silhouette_score, confusion_matrix

# Plotting Configuration
sns.set(style="whitegrid")

# ==========================================
# Section 1: Preprocessing
# ==========================================
print("--- Loading and Preprocessing Data ---")

# 1. Load Dataset
try:
    df = pd.read_csv('data/golestan.csv')
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'data/golestan.csv' not found.")
    exit()

# 2. Hazm Setup
normalizer = Normalizer()
lemmatizer = Lemmatizer()

# Define Stopwords
# Combine Hazm's default stopwords with frequent words in Golestan that have low semantic value for clustering
stop_words = set(stopwords_list())
custom_stops = {
    'گفت', 'گفتا', 'می‌گفت', 'بگفت', 'گویند', 'گفتم', 'گفتند', 
    'دید', 'دیدم', 'بدید', 'نظری', 'شنیدم', 'شنید',             
    'رفت', 'برفت', 'آمد', 'درآمد', 'بازآمد', 'آید', 'نیامد',    
    'یکی', 'کسی', 'کس', 'مردی', 'شخصی', 'بنده', 'مرا', 'او', 'من', 'تو',
    'بود', 'شد', 'گشت', 'گردید', 'است', 'نیست', 'دارد', 'کرد', 'کنند',
    'باشد', 'نباشد', 'شدی', 'بودی', 'گر', 'چو', 'افتاد', 'همی', 'ماند', 'داد',
    'کن', 'دانست', 'توان', 'باری', 'ای', 'که', 'از', 'به', 'در', 'را', 'با', 'و'
}
stop_words.update(custom_stops)

def clean_persian_text(text):
    """
    Cleans and preprocesses Persian text.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalization (fix spaces, half-spaces)
    text = normalizer.normalize(text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = []
    for t in tokens:
        if t not in stop_words and len(t) > 1:
            # Get lemma (root word)
            lemma = lemmatizer.lemmatize(t).split('#')[0]
            if lemma not in stop_words:
                cleaned_tokens.append(lemma)
                
    return " ".join(cleaned_tokens)

# Apply preprocessing
df['clean_text'] = df['hekayt'].apply(clean_persian_text)

# ==========================================
# Section 2: Vectorization (TF-IDF & LSA)
# ==========================================
print("\n--- Vectorization (TF-IDF & LSA) ---")

# 1. TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

# 2. Dimensionality Reduction (LSA)
# To reduce noise and improve clustering performance on short texts
n_components = 15
lsa = TruncatedSVD(n_components=n_components, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)

# 3. Normalization (L2 Normalization)
# To make Euclidean distance in K-Means behave like Cosine Similarity
normalizer_vec = SklearnNormalizer(norm='l2')
X_final = normalizer_vec.fit_transform(X_lsa)

print(f"Final Feature Matrix Shape: {X_final.shape}")

# Helper function to extract keywords for each cluster
def get_top_keywords(data_tfidf, labels, vectorizer, n_terms=5):
    df_temp = pd.DataFrame(data_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    df_temp['label'] = labels
    
    # Ignore noise points in DBSCAN (Label = -1)
    if -1 in df_temp['label'].values:
        df_temp = df_temp[df_temp['label'] != -1]

    cluster_means = df_temp.groupby('label').mean()
    
    print(f"\n--- Top {n_terms} Keywords per Cluster ---")
    for i, row in cluster_means.iterrows():
        top_indices = np.argsort(row)[-n_terms:][::-1]
        top_words = [row.index[idx] for idx in top_indices]
        print(f"Cluster {i:>2}: {', '.join(top_words)}")

# ==========================================
# Task 1: K-Means (K=5)
# ==========================================
print("\n" + "="*40)
print("   Task 1: K-Means with K=5")
print("="*40)

kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_5 = kmeans_5.fit_predict(X_final)
df['cluster_k5'] = labels_5

get_top_keywords(X_tfidf, labels_5, tfidf_vectorizer)

# ==========================================
# Task 2: Optimal K Analysis (Elbow & Silhouette)
# ==========================================
print("\n" + "="*40)
print("   Task 2: Optimal K Analysis")
print("="*40)

inertia = []
sil_scores = []
K_range = range(2, 15)

for k in tqdm(K_range, desc="Calculating Elbow/Silhouette"):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_final)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_final, km.labels_))

# Plot Elbow and Silhouette results
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color=color)
ax1.plot(K_range, inertia, 'bo-', label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(K_range, sil_scores, 'rs--', label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Optimal k Analysis')
plt.show()

# --- Run K-Means with K=11 (as requested) ---
optimal_k = 11
print(f"\nRunning K-Means with Optimal K={optimal_k}...")

kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
df['cluster_opt'] = kmeans_opt.fit_predict(X_final)

print("\nCluster Sizes (K=11):")
print(df['cluster_opt'].value_counts().sort_index())

# Plot Heatmap comparing with actual chapters (Babs)
crosstab_opt = pd.crosstab(df['cluster_opt'], df['bab'])
plt.figure(figsize=(10, 6))
sns.heatmap(crosstab_opt, annot=True, fmt='d', cmap='Greens')
plt.title(f'K-Means (K={optimal_k}) vs. Real Chapters (Babs)')
plt.ylabel('Cluster ID')
plt.xlabel('Real Chapter (Bab)')
plt.show()

get_top_keywords(X_tfidf, df['cluster_opt'], tfidf_vectorizer)

# ==========================================
# Task 3: DBSCAN Clustering
# ==========================================
print("\n" + "="*40)
print("   Task 3: DBSCAN Clustering")
print("="*40)

# Tuned parameters
eps = 0.64
min_samples = 3

print(f"Running DBSCAN (Eps={eps}, Min_Samples={min_samples})...")
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df['cluster_dbscan'] = dbscan.fit_predict(X_final)

# Cluster Statistics
n_clusters_db = len(set(df['cluster_dbscan'])) - (1 if -1 in df['cluster_dbscan'] else 0)
n_noise_db = list(df['cluster_dbscan']).count(-1)

print(f"Clusters Found: {n_clusters_db}")
print(f"Noise Points: {n_noise_db}")
print("\nCluster Sizes (DBSCAN):")
print(df['cluster_dbscan'].value_counts().sort_index())

# Plot Heatmap comparing DBSCAN clusters with actual chapters
crosstab_db = pd.crosstab(df['cluster_dbscan'], df['bab'])
plt.figure(figsize=(10, 6))
sns.heatmap(crosstab_db, annot=True, fmt='d', cmap='Reds')
plt.title('DBSCAN Clusters vs. Real Chapters (-1 is Noise)')
plt.ylabel('DBSCAN Cluster ID')
plt.xlabel('Real Chapter (Bab)')
plt.show()

get_top_keywords(X_tfidf, df['cluster_dbscan'], tfidf_vectorizer)

print("\nProcessing Complete.")