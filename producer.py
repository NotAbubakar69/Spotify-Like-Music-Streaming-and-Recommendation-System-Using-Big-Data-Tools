import random
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import scann
import librosa
from confluent_kafka import Producer
import json
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
# Kafka configuration
bootstrap_servers = 'localhost:9092'
topic = 'audio_recommendations'

# Load precomputed features from the CSV file
data_path = "/home/hdoop/Downloads/music_streaming_platform/preprocessed_audios.csv"  # Update this path if needed
features_df = pd.read_csv(data_path)

# Assume features are all columns after 'filename'
feature_columns = features_df.columns[1:448]
features = features_df[feature_columns].values
normalized_features = tf.math.l2_normalize(features, axis=1)

# Define ScannSearcherWrapper class for GridSearchCV
class ScannSearcherWrapper(BaseEstimator):
    def __init__(self, features, num_neighbors=10, num_leaves=10, num_leaves_to_search=5):
        self.features = features
        self.num_neighbors = num_neighbors
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        
    def fit(self, X, y=None):
        self.searcher = self.create_scann_index(self.features, self.num_neighbors, self.num_leaves, self.num_leaves_to_search)
        return self
    
    def create_scann_index(self, features, num_neighbors, num_leaves, num_leaves_to_search):
        searcher = scann.scann_ops_pybind.builder(
            features, num_neighbors, "dot_product"
        ).tree(
            num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search
        ).score_ah(
            2, anisotropic_quantization_threshold=0.2
        ).reorder(100).build()
        return searcher
    
    def search(self, X, n_neighbours=5):
        return self.get_nearest_neighbours(X, self.searcher, n_neighbours)
    
    def get_nearest_neighbours(self, feature_vector, searcher, n_neighbours=5):
        # Ensure the vector is one-dimensional and a numpy array
        if feature_vector.ndim != 1 or not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector).flatten()

        neighbors, distances = searcher.search(feature_vector, final_num_neighbors=n_neighbours)
        return neighbors, distances

# Define a grid of values for num_leaves
param_grid = {'num_leaves': [10, 20,30,40,50]}

grid_search = GridSearchCV(estimator=ScannSearcherWrapper(features=normalized_features), param_grid=param_grid, scoring='accuracy', cv=5)

# Perform grid search
grid_search.fit(normalized_features.numpy())

# Get the best hyperparameters
best_num_leaves = grid_search.best_params_['num_leaves']

# Update the create_scann_index function with the best num_leaves
def create_scann_index(features, num_neighbors=10, num_leaves=best_num_leaves, num_leaves_to_search=5):
    searcher = scann.scann_ops_pybind.builder(
        features, num_neighbors, "dot_product"
    ).tree(
        num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search
    ).score_ah(
        2, anisotropic_quantization_threshold=0.2
    ).reorder(100).build()
    return searcher

# Create the ScaNN searcher with the best num_leaves
scann_searcher = create_scann_index(normalized_features)

def load_and_query(audio_path, features_df, searcher, n_neighbours=5):
    if not os.path.exists(audio_path):
        print("File not found. Please check the path and try again.")
        return None, None

    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=50)
    spectrogram = librosa.amplitude_to_db(S, ref=np.max)
    spectrogram_mean = spectrogram.mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempogram_mean = tempogram.mean(axis=1)

    feature_vector = np.hstack([mfcc_mean, spectrogram_mean, tempogram_mean])
    if np.linalg.norm(feature_vector) > 0:
        feature_vector_normalized = feature_vector / np.linalg.norm(feature_vector)
    else:
        feature_vector_normalized = feature_vector

    nearest_neighbours, distances = get_nearest_neighbours(feature_vector_normalized, searcher, n_neighbours)
    return nearest_neighbours, distances

def get_nearest_neighbours(feature_vector, searcher, n_neighbours=5):
    # Ensure the vector is one-dimensional and a numpy array
    if feature_vector.ndim != 1 or not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector).flatten()

    neighbors, distances = searcher.search(feature_vector, final_num_neighbors=n_neighbours)
    return neighbors, distances

def extract_audio_features(audio_files_path):
    audio_files = []
    for root, dirs, files in os.walk(audio_files_path):
        for file in files:
            if file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    if not audio_files:
        print("No audio files found in the specified directory.")
        return None
    random_audio_path = random.choice(audio_files)
    return random_audio_path

# Randomly select an audio file from the dataset
random_audio_path = extract_audio_features("/home/mutayyab/Documents/dataset")


def produce_recommendations(producer, features_df, searcher, num_recommendations=5):
    # Get the name of the randomly chosen audio file with the .mp3 extension
    random_audio_name = os.path.basename(random_audio_path)

    # Load and query the randomly selected audio file
    nearest_neighbours, distances = load_and_query(random_audio_path, features_df, searcher, num_recommendations)
    
    if nearest_neighbours is not None and distances is not None:
        # Produce messages containing the IDs or paths of the similar audio files
        recommendations = []
        for neighbour, distance in zip(nearest_neighbours, distances):
            recommendation = {
                'randomly_chosen_file': random_audio_name,  # Add this line to include the randomly chosen file name
                'recommended_track': features_df.iloc[neighbour]['filename'],
                'similarity_score': 1 - distance
            }
            recommendations.append(recommendation)
        
        # Sort recommendations in descending order based on similarity score
        sorted_recommendations = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
        
        # Print sorted recommendations
        for rec in sorted_recommendations:
            print("Recommended Track:", rec['recommended_track'])
            print("Similarity Score:", rec['similarity_score'])
            print()
        
        # Convert recommendations to JSON and produce to Kafka topic
        producer.produce(topic, json.dumps(sorted_recommendations))
        producer.flush()



# Kafka producer configuration
config = {
    'bootstrap.servers': bootstrap_servers
}

# Create Kafka producer
producer = Producer(config)

# Produce recommendation messages using a randomly selected audio file
produce_recommendations(producer, features_df, scann_searcher)
