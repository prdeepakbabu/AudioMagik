#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 21:36:18 2025
@author: badeepak
"""

import time
start_time = time.time()

import torch
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, HubertModel
from datasets import Dataset, Audio
# Auto-detect device (CUDA for Nvidia, MPS for Apple Metal, or CPU fallback)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

############################################
# 1. Create a Hugging Face Dataset with local MP3 path(s)
############################################

# Example: single MP3 file
data = {"audio": ["/Users/badeepak/Downloads/audio.mp3"]}
data = {"audio": ["/Users/badeepak/Downloads/Arthur.mp3"]}

# Build dataset from a Python dictionary
dataset = Dataset.from_dict(data)

# "cast_column" transforms the "audio" column into an Audio feature (decodes MP3 automatically)
dataset = dataset.cast_column("audio", Audio(decode=True))

# Now "dataset[0]['audio']" is a dictionary with keys "array" and "sampling_rate"
audio_info = dataset[0]["audio"]
waveform = audio_info["array"]  # a NumPy float32 array

# 1) Convert to PyTorch tensor
waveform = torch.from_numpy(waveform).float()

# 2) Make sure shape is (channels, time)
if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)

orig_sample_rate = audio_info["sampling_rate"]  # an integer

if orig_sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sample_rate,
        new_freq=16000
    )
    waveform = resampler(waveform)  # Now it's at 16kHz
    sample_rate = 16000


############################################
# 2. Visualize a Spectrogram
############################################

def visualize_spectrogram(waveform, sample_rate, title="Spectrogram"):
    """
    Plots the mel spectrogram of a waveform.
    waveform: 1D (mono) or 2D array (channels x time)
    sample_rate: Audio sample rate
    """
    # If the waveform is a NumPy array, convert to Torch tensor
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # shape: (1, time)

    # Create the mel spectrogram
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    mel = mel_spectrogram(waveform_tensor)  # shape: (1, 128, time_frames)

    # Convert to decibels
    mel_db = mel.log2()[0].numpy()  # shape: (128, time_frames)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.show()


visualize_spectrogram(audio_info["array"], sample_rate, title="Spectrogram of MP3 Audio")

############################################
# 3. Tokenize Audio with HuBERT
############################################

# Load the HuBERT processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


def tokenize_audio(waveform, sample_rate):
    """
    Runs HuBERT model on audio and returns the last hidden state (token embeddings).
    """
    # Move waveform tensor to the correct device
    waveform = waveform.to(device)

    # Prepare input for HuBERT
    inputs = processor(
        waveform.squeeze().cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    print(inputs)

    # Forward pass through HuBERT
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)

    return outputs.last_hidden_state  # shape: (batch_size=1, time_frames, hidden_dim=1024)


tokens = tokenize_audio(waveform, sample_rate)

print("Tokenized Audio Shape:", tokens.shape)
# e.g. torch.Size([1, T, 1024])

######################


import torch
from transformers import Wav2Vec2Processor, HubertForCTC

# Load the HuBERT processor and fine-tuned ASR model
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


def transcribe_audio(waveform, sample_rate):
    """
    Runs HuBERT model on audio and returns transcribed text.
    """
    # Prepare input for HuBERT
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # Forward pass through HuBERT with CTC head
    with torch.no_grad():
        logits = model(**inputs).logits  # Shape: (batch_size=1, time_steps, vocab_size)

    # Decode logits to characters
    predicted_ids = torch.argmax(logits, dim=-1)  # Get the highest probability token at each time step
    transcription = processor.batch_decode(predicted_ids)[0]  # Convert token IDs to text

    return transcription


# Transcribe the audio
transcription = transcribe_audio(waveform, sample_rate)

print("Transcription:", transcription)


# Load HuBERT ASR model & processor
from transformers import Wav2Vec2Processor, HubertForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)  # Move model to GPU


def transcribe_audio(waveform, sample_rate):
    """
    Runs HuBERT model on audio and returns transcribed text.
    """
    # Move waveform tensor to the correct device
    waveform = waveform.to(device)

    # Prepare input for HuBERT
    inputs = processor(
        waveform.squeeze().cpu().numpy(),  # Convert to NumPy (needed for processor)
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # Move input tensors to GPU if available
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Forward pass through HuBERT with CTC head
    with torch.no_grad():
        logits = model(**inputs).logits  # Shape: (batch_size=1, time_steps, vocab_size)

    # Decode logits to characters
    predicted_ids = torch.argmax(logits, dim=-1)  # Get highest probability token at each time step
    transcription = processor.batch_decode(predicted_ids)[0]  # Convert token IDs to text

    return transcription


# Transcribe the audio
transcription = transcribe_audio(waveform, sample_rate)

print("Transcription:", transcription)

from transformers import HubertForCTC, Wav2Vec2Processor
import torch
import joblib

# Load HuBERT with the quantizer
# Convert embeddings to NumPy (for K-Means clustering)
embeddings_np = tokens.squeeze(0).cpu().numpy()  # Shape: (Time, 1024)

from sklearn.cluster import KMeans

# Perform K-Means clustering on the embeddings with 100 clusters
kmeans = KMeans(n_clusters=100, random_state=42)
kmeans.fit(embeddings_np)
hubert_tokens = kmeans.predict(embeddings_np)

print("HuBERT Token IDs:", hubert_tokens[:10])  # Print first 10 tokenized values
print("Tokenized Shape:", hubert_tokens.shape)  # Expected: (Time,)


def align_tokens_to_phonemes(hubert_tokens, phoneme_transcription):
    """
    Align HuBERT tokens with a sequence of phonemes using a simple linear mapping.

    Parameters:
    - hubert_tokens: list or numpy array of HuBERT tokens.
    - phoneme_transcription: string containing phonemes separated by whitespace,
                             e.g., "AH K AE T".

    Returns:
    - A list where each HuBERT token is mapped to a phoneme.
    """
    # Convert the transcription string into a list of phonemes
    phonemes = phoneme_transcription.split()

    num_tokens = len(hubert_tokens)
    num_phonemes = len(phonemes)

    aligned_phonemes = []
    for i in range(num_tokens):
        # Compute the phoneme index for this token by linear interpolation
        phoneme_idx = int(i * num_phonemes / num_tokens)
        # Ensure the index does not exceed bounds
        if phoneme_idx >= num_phonemes:
            phoneme_idx = num_phonemes - 1
        aligned_phonemes.append(phonemes[phoneme_idx])

    return aligned_phonemes


# Example usage:
# Assume hubert_tokens is already computed, for instance, using K-Means clustering.
# Here, we simulate hubert_tokens as a list of 100 tokens for demonstration.
import numpy as np

#hubert_tokens = np.arange(100)  # Replace with your actual HuBERT token IDs

# Assume you have a phoneme transcription (this should be generated via forced alignment in practice)
#phoneme_transcription = "AH K AE T"  # Replace with your actual phoneme transcription

import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

def text_to_phonemes(text):
    d = cmudict.dict()
    words = text.lower().split()
    phonemes_list = []
    for word in words:
        if word in d:
            phonemes_list.append(d[word][0])
        else:
            phonemes_list.append(["UNK"])
    return phonemes_list

text = "THE STORY OF ARTHUR THE RAT ONCE UPON A TIME ..."
phoneme_transcription = text_to_phonemes(transcription)
print(phoneme_transcription)

# Align the tokens to the phonemes
aligned_mapping = align_tokens_to_phonemes(hubert_tokens, phoneme_transcription)
print("Aligned Phoneme Mapping for first 10 tokens:", aligned_mapping[:10])

end_time = time.time()
print(f"Total script runtime: {end_time - start_time:.2f} seconds")