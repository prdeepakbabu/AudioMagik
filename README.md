# AudioMagik

AudioMagik is a comprehensive audio processing pipeline that demonstrates end-to-end audio analysis using deep learning models. It covers everything from loading and preprocessing audio files to visualizing spectrograms, tokenizing and transcribing audio using the HuBERT model, and discretizing token embeddings with K-Means clustering.

## Features

- **Audio Loading & Preprocessing:**  
  Load local MP3 files using Hugging Face's `datasets` library and preprocess them by resampling if necessary.
  
- **Spectrogram Visualization:**  
  Convert audio waveforms into mel spectrograms using `torchaudio.transforms.MelSpectrogram` and visualize them with Matplotlib.
  
- **Tokenization & Transcription:**  
  Leverage a pre-trained HuBERT model (`facebook/hubert-large-ls960-ft`) to tokenize and transcribe audio, with support for both CPU and GPU (including Apple's MPS).
  
- **Discretization of Token Embeddings:**  
  Apply a pre-trained K-Means clustering model to convert continuous token embeddings into discrete tokens.
  
- **Runtime Timing:**  
  Measure and print the total script runtime.

## Requirements

Ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Install Dependencies:**  
   Run the command above to install all required packages.
   
2. **Run the Script:**  
   Execute the main script with:
   ```bash
   python main.py
   ```
   The script will:
   - Load and preprocess an audio file.
   - Visualize its spectrogram.
   - Tokenize and transcribe the audio using the HuBERT model.
   - Discretize the token embeddings with K-Means clustering.
   - Print the total runtime of the script.

3. **Device Support:**  
   The script auto-detects available devices and prefers:
   - CUDA (if available)
   - MPS (for Apple devices)
   - CPU (fallback)

## File Structure

- **`main.py`**: Main script containing the audio processing pipeline.
- **`requirements.txt`**: List of required Python packages.
- **`README.md`**: This documentation file.
- Additional files may include model weights or clustering data if applicable.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Torchaudio](https://github.com/pytorch/audio)
- [PyTorch](https://github.com/pytorch/pytorch)
- Additional thanks to the open-source community for their contributions to these libraries.