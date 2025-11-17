"""
Download and prepare the Shakespeare dataset for character-level language modeling.
"""
import os
import requests


def download_shakespeare():
    """Download the Tiny Shakespeare dataset."""
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/shakespeare.txt'
    
    # Check if already downloaded
    if os.path.exists(output_path):
        print(f"[INFO] Dataset already exists at {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[INFO] Length: {len(text):,} characters")
        return text
    
    # Download
    print(f"[INFO] Downloading Shakespeare dataset from {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    text = response.text
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"[INFO] Downloaded {len(text):,} characters to {output_path}")
    return text


def load_shakespeare():
    """Load the Shakespeare dataset, downloading if necessary."""
    output_path = 'data/shakespeare.txt'
    
    if not os.path.exists(output_path):
        return download_shakespeare()
    
    with open(output_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"[INFO] Loaded {len(text):,} characters from {output_path}")
    return text


def prepare_data(text, train_split=0.9):
    """
    Split text into train and validation sets.
    
    Args:
        text: Full text string
        train_split: Fraction for training (default 0.9)
    
    Returns:
        train_text, val_text
    """
    n = len(text)
    train_size = int(n * train_split)
    
    train_text = text[:train_size]
    val_text = text[train_size:]
    
    print(f"[INFO] Train size: {len(train_text):,} characters")
    print(f"[INFO] Val size: {len(val_text):,} characters")
    
    return train_text, val_text


if __name__ == '__main__':
    # Test the download
    text = download_shakespeare()
    train, val = prepare_data(text)
    
    print("\nFirst 100 characters:")
    print(train[:100])
