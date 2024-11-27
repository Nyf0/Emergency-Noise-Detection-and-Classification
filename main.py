#!/usr/bin/env python
# coding: utf-8

# In[53]:


# Importing Necessary Libraries

# Audio Processing Libraries
import librosa
import librosa.display
from scipy import signal

# For Playing Audios
import IPython.display as ipd

# Array Processing
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display the confusion matrix
from sklearn.metrics import confusion_matrix

# Create a DataFrame
import pandas as pd

import pickle

# Encode Categorical Targets
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Split Dataset
from sklearn.model_selection import train_test_split

import os


# ## Exploring the dataset
# 
# # Setting directories for training and testing datasets

# In[54]:


# Set Dataset Paths
dataset_path = './dataset/'
training_path = os.path.join(dataset_path, 'training/')
testing_path = os.path.join(dataset_path, 'testing/')


# # Function to Plot and Play Audio

# In[55]:


def display_audio_info(file_path):
    plt.figure(figsize=(14, 5))
    data, sample_rate = librosa.load(file_path)
    librosa.display.waveshow(data, sr=sample_rate)
    plt.title(f"Waveform of {os.path.basename(file_path)}")
    plt.show()
    return ipd.Audio(file_path)


# # Display Example from Ambulance Class

# In[56]:


ambulance_example = os.path.join(training_path, 'ambulance', 'sound_1.wav')
display_audio_info(ambulance_example)


# # Display Example from Firetruck Class

# In[57]:


firetruck_example = os.path.join(training_path, 'firetruck', 'sound_202.wav')
display_audio_info(firetruck_example)


# # Display Example from Traffic Class

# In[58]:


traffic_example = os.path.join(training_path, 'traffic', 'sound_405.wav')
display_audio_info(traffic_example)


# # Feature extraction function

# In[59]:


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# Show extracted features

# In[60]:


extracted_features = []
for split in ['training', 'testing']:
    for label in ['ambulance', 'firetruck', 'traffic']:
        folder_path = os.path.join(dataset_path, split, label)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                data = features_extractor(file_path)
                extracted_features.append([data, label])


# Save the extrcted features

# In[61]:


with open('./Extracted_Features.pkl', 'wb') as f:
    pickle.dump(extracted_features, f)


# Loading extracted features

# In[62]:


with open('./Extracted_Features.pkl', 'rb') as f:
    data = pickle.load(f)


# Converting data frame

# In[63]:


df = pd.DataFrame(data, columns=['feature', 'class'])
print(df.head())
print(df['class'].value_counts())


# # Train-Test split

# In[64]:


#Split Data into Train/Test Sets
X = np.array(df['feature'].tolist())
Y = np.array(df['class'].tolist())

#Label Encoding
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(Y))

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True
)


# Display dataset shape

# In[65]:


# Display Dataset Shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# ## CNN Model

# In[66]:


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report

def cnn(optimizer="adam", activation="relu", dropout_rate=0.5):
    K.clear_session()
    inputs = Input(shape=(X_train.shape[1], 1))
    conv = Conv1D(64, kernel_size=3, padding='same', activation=activation)(inputs)
    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = GlobalMaxPooling1D()(conv)
    conv = Dense(32, activation=activation)(conv)
    outputs = Dense(y_train.shape[1], activation='softmax')(conv)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Prepare features

# In[67]:


X_train_features = X_train.reshape(len(X_train), -1, 1)
X_test_features = X_test.reshape(len(X_test), -1, 1)


# # Training

# In[68]:


model_cnn = cnn(optimizer="adam", activation="relu", dropout_rate=0.3)
history = model_cnn.fit(
    X_train_features,
    y_train,
    validation_data=(X_test_features, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)


# # Training History

# In[69]:


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Model Evaluation

# In[70]:


_, acc = model_cnn.evaluate(X_test_features, y_test)
print(f"Test Accuracy: {acc}")

# Classification Report
y_pred = np.argmax(model_cnn.predict(X_test_features), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=labelencoder.classes_))


# # Confusion matrix

# In[71]:


conf_mat = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labelencoder.classes_, yticklabels=labelencoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Saving classification results to csv

# In[76]:


# Function to save classification results to respective folder
def save_results_to_csv(folder_path, y_true, y_pred, file_names, labelencoder, output_file_name):
    """
    Saves classification results to a CSV file in the specified folder.
    
    Parameters:
    - folder_path (str): Path to the folder where the CSV will be saved.
    - y_true (array): Ground truth labels (one-hot encoded).
    - y_pred (array): Predicted probabilities from the model.
    - file_names (list): List of sound file names.
    - labelencoder (LabelEncoder): Label encoder for mapping numeric labels to class names.
    - output_file_name (str): Name of the output CSV file.
    """
    # Convert one-hot encoded labels to categorical indices
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_confidence = np.max(y_pred, axis=1)  # Confidence score

    # Decode labels to class names
    actual_labels = labelencoder.inverse_transform(y_true_labels)
    predicted_labels = labelencoder.inverse_transform(y_pred_labels)

    # Extract sound numbers from file names (assuming file names are like 'sound_123.wav')
    sound_numbers = [os.path.splitext(file)[0] for file in file_names]

    # Ensure all lists have the same length
    if len(sound_numbers) == len(actual_labels) == len(predicted_labels) == len(y_pred_confidence):
        # Create a DataFrame
        results_df = pd.DataFrame({
            "sound_number": sound_numbers,
            "actual": actual_labels,
            "predicted": predicted_labels,
            "confidence_score": y_pred_confidence
        })

        # Save the DataFrame as a CSV file in the specified folder
        output_path = os.path.join(folder_path, output_file_name)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("Error: Lists have different lengths.")
        print(f"Lengths - sound_numbers: {len(sound_numbers)}, actual_labels: {len(actual_labels)}, predicted_labels: {len(predicted_labels)}, confidence_score: {len(y_pred_confidence)}")


# Example: Save Training Results
y_pred_train = model_cnn.predict(X_train_features)
print("Training Predictions:", y_pred_train)
print("Training True Labels:", y_train)

# Generate the list of training file names
train_file_names = []

# Add files from sound_1.wav to sound_160.wav
train_file_names.extend([f"sound_{i}.wav" for i in range(1, 161)])

# Add files from sound_200.wav to sound_360.wav
train_file_names.extend([f"sound_{i}.wav" for i in range(200, 361)])

# Add files from sound_400.wav to sound_560.wav
train_file_names.extend([f"sound_{i}.wav" for i in range(400, 561)])

save_results_to_csv(
    folder_path="./dataset/training/",
    y_true=y_train,
    y_pred=model_cnn.predict(X_train_features),
    file_names=train_file_names,
    labelencoder=labelencoder,
    output_file_name="training.csv"
)

# Generate the list of training file names
test_file_names = []

# Add files from sound_1.wav to sound_160.wav
test_file_names.extend([f"sound_{i}.wav" for i in range(161, 201)])

# Add files from sound_200.wav to sound_360.wav
test_file_names.extend([f"sound_{i}.wav" for i in range(201, 401)])

# Add files from sound_400.wav to sound_560.wav
test_file_names.extend([f"sound_{i}.wav" for i in range(401, 601)])

save_results_to_csv(
    folder_path="./dataset/testing/",
    y_true=y_test,
    y_pred=model_cnn.predict(X_test_features),
    file_names=test_file_names,
    labelencoder=labelencoder,
    output_file_name="testing.csv"
)

