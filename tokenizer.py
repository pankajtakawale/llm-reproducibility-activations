"""
Character-level tokenizer for the Shakespeare dataset.
Simple mapping from characters to integers and vice versa.
"""


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text):
        """
        Initialize tokenizer with vocabulary from text.
        
        Args:
            text: String to build vocabulary from
        """
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"[INFO] Vocabulary size: {self.vocab_size}")
        print(f"[INFO] Characters: {''.join(chars)}")
    
    def encode(self, text):
        """
        Convert text to list of integers.
        
        Args:
            text: String to encode
        
        Returns:
            List of integers
        """
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """
        Convert list of integers back to text.
        
        Args:
            indices: List of integers or tensor
        
        Returns:
            String
        """
        # Handle tensors
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        return ''.join([self.idx_to_char[i] for i in indices])
    
    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size


if __name__ == '__main__':
    # Test the tokenizer
    from prepare_data import load_shakespeare
    
    text = load_shakespeare()
    tokenizer = CharTokenizer(text)
    
    # Test encoding/decoding
    sample = "Hello, World!"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {sample == decoded}")
