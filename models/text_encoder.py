"""
Text encoder implementations for conditioning sprite generation.

This module provides text encoding methods to convert text descriptions
into embedding vectors that can condition the Generator.
"""

import torch
import torch.nn as nn

from spriteforge.models.base import BaseTextEncoder


class SimpleTextEncoder(BaseTextEncoder):
    """
    Simple word-embedding based text encoder.
    
    Uses a vocabulary-based approach with learned word embeddings
    and aggregation (mean pooling) to create sentence embeddings.
    
    This is lightweight and suitable for simple text descriptions
    with limited vocabulary (e.g., "red warrior", "blue potion").
    
    Attributes:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of word embeddings.
        padding_idx: Index for padding tokens.
    
    Example:
        >>> encoder = SimpleTextEncoder(vocab_size=1000, embedding_dim=256)
        >>> text = ["red warrior", "blue potion"]
        >>> embeddings = encoder(text)  # shape: (2, 256)
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 256,
        padding_idx: int = 0,
        max_length: int = 32,
    ) -> None:
        """
        Initialize the simple text encoder.
        
        Args:
            vocab_size: Number of words in vocabulary.
            embedding_dim: Dimension of text embeddings.
            padding_idx: Index used for padding.
            max_length: Maximum sequence length.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_length = max_length
        
        # Word embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # Optional: Simple LSTM/GRU for sequential processing
        self.lstm = nn.LSTM(
            embedding_dim,
            embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        
        # Projection layer to final embedding dimension
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            self.embedding.weight.data[self.padding_idx].zero_()
        
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text_indices: Token indices of shape (batch, seq_len).
            
        Returns:
            Text embeddings of shape (batch, embedding_dim).
        """
        # Embed tokens
        embedded = self.embedding(text_indices)  # (batch, seq_len, embed_dim)
        
        # Process with LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state (concatenated forward and backward)
        # hidden shape: (num_layers * 2, batch, hidden_dim//2)
        # Take last layer, concatenate both directions
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, embed_dim)
        
        # Project to final embedding
        output = self.projection(final_hidden)
        
        return output
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the text embeddings."""
        return self.embedding_dim


class CLIPStyleEncoder(BaseTextEncoder):
    """
    CLIP-inspired text encoder using transformer architecture.
    
    More powerful than SimpleTextEncoder, suitable for complex
    descriptions and better text-image alignment.
    
    Note: This is a simplified version. For production, consider
    using actual CLIP from OpenAI or similar pretrained models.
    
    Attributes:
        embedding_dim: Dimension of text embeddings.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
    
    Example:
        >>> encoder = CLIPStyleEncoder(embedding_dim=512, num_heads=8, num_layers=4)
        >>> text_indices = torch.randint(0, 1000, (4, 20))  # (batch=4, seq_len=20)
        >>> embeddings = encoder(text_indices)  # shape: (4, 512)
    """
    
    def __init__(
        self,
        vocab_size: int = 49408,  # CLIP vocab size
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_length: int = 77,  # CLIP max length
        padding_idx: int = 0,
    ) -> None:
        """
        Initialize the CLIP-style encoder.
        
        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            max_length: Maximum sequence length.
            padding_idx: Padding token index.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.positional_embedding = nn.Parameter(torch.randn(max_length, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.ln_final = nn.LayerNorm(embedding_dim)
        
        # Projection head for final embedding
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.projection.weight, std=self.embedding_dim ** -0.5)
    
    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text_indices: Token indices of shape (batch, seq_len).
            
        Returns:
            Text embeddings of shape (batch, embedding_dim).
        """
        batch_size, seq_len = text_indices.shape
        
        # Token embeddings
        x = self.token_embedding(text_indices)  # (batch, seq_len, embed_dim)
        
        # Add positional embeddings
        x = x + self.positional_embedding[:seq_len]
        
        # Create attention mask (ignore padding)
        # True means "ignore this position"
        padding_mask = (text_indices == 0)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Use embedding of [EOS] token or mean pooling
        # For simplicity, use mean pooling over non-padded tokens
        mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, embed_dim)
        
        # Final layer norm and projection
        x = self.ln_final(x)
        x = self.projection(x)
        
        # L2 normalize for cosine similarity (CLIP-style)
        x = x / x.norm(dim=-1, keepdim=True)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the text embeddings."""
        return self.embedding_dim
