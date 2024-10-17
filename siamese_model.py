import glob
import tensorflow as tf
import pathlib
import numpy as np
import os
import random
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras import layers, models, Input
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

# For reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# Step 1: Set up Kaggle API credentials and download the dataset
def download_kaggle_dataset(data_dir):
    # Check if the dataset directory exists and contains images
    if os.path.exists(data_dir) and list(pathlib.Path(data_dir).glob('*/*.jpg')):
        print("Dataset already exists. Skipping download.")
        return data_dir

    kaggle_json_path = os.path.join(os.path.dirname(__file__), 'kaggle', 'kaggle.json')
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError("kaggle.json not found. Please place it in ~/.kaggle/")

    # Set permissions
    os.chmod(kaggle_json_path, 0o600)

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define the dataset
    dataset = 'vishesh1412/celebrity-face-image-dataset'  # Update with the correct dataset identifier

    # Define the download directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download and unzip the dataset
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset, path=str(data_dir), unzip=True)
    print("Download completed.")

    return data_dir
def load_images(data_dir):
    # Use glob to find images in the directory and its subdirectories
    image_files = glob.glob(str(data_dir / '**' / '*.jpg'), recursive=True)
    return image_files


# Step 3: Create triplets for training and testing
def create_triplets(image_paths, label_encoder, split='train', test_size=0.2, train_size=0.8):
    triplets = []
    label_dict = {}

    # Encode labels
    labels = [label_encoder.fit_transform([img_path.parent.name])[0] for img_path in image_paths]

    # Create a label dictionary for each class
    for img_path, label in zip(image_paths, labels):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(img_path)

    # Remove labels with less than 2 images
    label_dict = {label: imgs for label, imgs in label_dict.items() if len(imgs) >= 2}

    # Split labels into training and testing
    train_labels, test_labels = train_test_split(
        list(label_dict.keys()), test_size=test_size,train_size=train_size, random_state=42
    )

    if split == 'train':
        selected_labels = train_labels
    else:
        selected_labels = test_labels

    # Generate triplets
    for label in selected_labels:
        imgs = label_dict[label]
        for anchor in imgs:
            positive = random.choice(imgs)
            while positive == anchor:
                positive = random.choice(imgs)

            negative_label = random.choice([lbl for lbl in label_dict.keys() if lbl != label])
            negative = random.choice(label_dict[negative_label])

            triplets.append((anchor, positive, negative))

    return triplets


# Step 4: Data preprocessing function
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(105, 105))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img


# Step 5: Create a data generator for the triplets
class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, triplets, batch_size=32, shuffle=True):
        self.triplets = triplets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.triplets) // self.batch_size

    def __getitem__(self, idx):
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchor_images = np.array([preprocess_image(t[0]) for t in batch_triplets])
        positive_images = np.array([preprocess_image(t[1]) for t in batch_triplets])
        negative_images = np.array([preprocess_image(t[2]) for t in batch_triplets])
        return [anchor_images, positive_images, negative_images], np.zeros((self.batch_size,))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.triplets)


# Step 6: Define the Siamese Network architecture
def build_base_network(input_shape):
    base_model = models.Sequential([
        layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='sigmoid')  # Embedding size of 64
    ])
    return base_model


def build_siamese_model(input_shape):
    base_network = build_base_network(input_shape)

    input_anchor = Input(shape=input_shape, name='anchor_input')
    input_positive = Input(shape=input_shape, name='positive_input')
    input_negative = Input(shape=input_shape, name='negative_input')

    embedding_anchor = base_network(input_anchor)
    embedding_positive = base_network(input_positive)
    embedding_negative = base_network(input_negative)

    # Concatenate the embeddings for the triplet loss
    merged_output = layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    model = models.Model(inputs=[input_anchor, input_positive, input_negative], outputs=merged_output)
    return model


# Step 7: Implement triplet loss
def triplet_loss(y_true, y_pred, alpha=0.2):
    # Assuming embeddings are concatenated: [anchor, positive, negative]
    anchor = y_pred[:, 0:64]
    positive = y_pred[:, 64:128]
    negative = y_pred[:, 128:192]

    # Compute the pairwise distance
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # Compute the triplet loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.mean(K.maximum(basic_loss, 0.0), axis=0)
    return loss


# Step 8: Visualization Functions
def visualize_sample_triplets(triplets, label_encoder, num_samples=5):
    plt.figure(figsize=(10, num_samples * 3))
    for i in range(num_samples):
        anchor, positive, negative = triplets[i]

        # Decode labels
        anchor_label = label_encoder.inverse_transform([int(anchor.parent.name.split('_')[-1])])[0]
        positive_label = label_encoder.inverse_transform([int(positive.parent.name.split('_')[-1])])[0]
        negative_label = label_encoder.inverse_transform([int(negative.parent.name.split('_')[-1])])[0]

        # Plot anchor
        img = preprocess_image(anchor)
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(img)
        plt.title(f"Anchor: {anchor_label}")
        plt.axis('off')

        # Plot positive
        img = preprocess_image(positive)
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(img)
        plt.title(f"Positive: {positive_label}")
        plt.axis('off')

        # Plot negative
        img = preprocess_image(negative)
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(img)
        plt.title(f"Negative: {negative_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def visualize_embeddings(model, image_paths, label_encoder, num_samples=1000):
    # Select a subset for visualization
    selected_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    images = np.array([preprocess_image(p) for p in selected_paths])
    labels = [label_encoder.transform([int(p.parent.name.split('_')[-1])])[0] for p in selected_paths]

    # Create embedding model
    embedding_model = models.Model(inputs=model.input[0],
                                   outputs=model.get_layer(index=6).output)  # Adjust layer index as needed

    embeddings = embedding_model.predict(images, batch_size=64)

    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


# Step 9: Evaluation Function
def evaluate_model(model, triplets, label_encoder, threshold=1.0):
    correct = 0
    total = len(triplets)
    base_network = model.get_layer(index=3)  # Adjust based on model architecture

    for triplet in triplets:
        anchor_img = np.expand_dims(preprocess_image(triplet[0]), axis=0)
        positive_img = np.expand_dims(preprocess_image(triplet[1]), axis=0)
        negative_img = np.expand_dims(preprocess_image(triplet[2]), axis=0)

        # Get embeddings
        anchor_emb = base_network.predict(anchor_img)
        positive_emb = base_network.predict(positive_img)
        negative_emb = base_network.predict(negative_img)

        # Compute distances
        pos_dist = np.sum((anchor_emb - positive_emb) ** 2)
        neg_dist = np.sum((anchor_emb - negative_emb) ** 2)

        if pos_dist + threshold < neg_dist:
            correct += 1

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Specify your dataset directory
    dataset_directory = pathlib.Path(
        '/home/titus/Documents/siamese_model/data')  # Change this to your actual dataset path

    # Ensure the directory exists
    dataset_directory.mkdir(parents=True, exist_ok=True)

    # Download the dataset if it doesn't exist
    dataset_directory = download_kaggle_dataset(dataset_directory)

    # Load image paths
    image_paths = load_images(dataset_directory)

    if not image_paths:
        raise ValueError("No images found in the dataset directory.")


    # Encode labels
    label_encoder = LabelEncoder()
    labels = [img_path.parent.name for img_path in image_paths]
    label_encoder.fit(labels)

    # Split into training and testing triplets
    train_triplets = create_triplets(image_paths, label_encoder, split='train', test_size=0.2)
    test_triplets = create_triplets(image_paths, label_encoder, split='test', test_size=0.2)

    if not train_triplets:
        raise ValueError("No training triplets could be created. Check dataset labels and sizes.")

    if not test_triplets:
        raise ValueError("No testing triplets could be created. Check dataset labels and sizes.")

    print(f"Number of training triplets: {len(train_triplets)}")
    print(f"Number of testing triplets: {len(test_triplets)}")

    # Visualize some sample triplets
    visualize_sample_triplets(train_triplets, label_encoder, num_samples=3)

    # Define model parameters
    input_shape = (105, 105, 3)
    siamese_model = build_siamese_model(input_shape)
    siamese_model.compile(loss=triplet_loss, optimizer='adam')

    # Define training parameters
    batch_size = 32
    epochs = 1  # Increased epochs for better training

    # Create data generators
    train_generator = TripletDataGenerator(train_triplets, batch_size=batch_size, shuffle=True)
    test_generator = TripletDataGenerator(test_triplets, batch_size=batch_size, shuffle=False)

    # Train the model
    history = siamese_model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator
    )

    # Plot training loss
    plot_training_loss(history)

    # Visualize embeddings
    visualize_embeddings(siamese_model, image_paths, label_encoder, num_samples=1000)

    # Evaluate the model
    evaluate_model(siamese_model, test_triplets, label_encoder, threshold=1.0)

    # Save the model (optional)
    model_save_path = '/models/siamese_model.h5'
    siamese_model.save(model_save_path)
    print(f"Model training completed and saved at {model_save_path}.")
