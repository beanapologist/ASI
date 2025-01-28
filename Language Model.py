import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Updated imports: Removed deprecated prepare_dynamic and convert_dynamic_fx
from torch.quantization import quantize_dynamic #, prepare_dynamic, convert_dynamic_fx
# Removed unused import
# from torch.quantization.quantize_fx import prepare_fx, convert_fx # prepare_fx, convert_fx
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import os
from tqdm import tqdm

@dataclass
class QuantizedLMConfig:
    """Configuration for Quantized Quantum Language Model"""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    vocab_size: int = 30522
    wormhole_hidden_size: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    quantization_dtype: torch.dtype = torch.qint8
    quantization_scheme: str = 'dynamic'  # 'dynamic' or 'static'

class QuantizableWormholeAttention(nn.Module):
    """Quantization-friendly wormhole attention"""
    def __init__(self, config: QuantizedLMConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} not divisible by number of attention heads {config.num_attention_heads}"
            )
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Quantizable linear layers
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Wormhole gate - using ReLU for better quantization
        self.wormhole_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.wormhole_hidden_size),
            nn.LayerNorm(config.wormhole_hidden_size, eps=config.layer_norm_eps),
            nn.ReLU(),  # Changed from GELU for better quantization
            nn.Linear(config.wormhole_hidden_size, self.num_attention_heads),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.size()[:2]

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        wormhole_gates = self.wormhole_gate(hidden_states)
        wormhole_gates = wormhole_gates.permute(0, 2, 1).unsqueeze(-1)
        attention_scores = attention_scores * wormhole_gates

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        output = self.layer_norm(output + hidden_states)

        return output, attention_probs

class QuantizableTransformerLayer(nn.Module):
    """Quantization-friendly transformer layer"""
    def __init__(self, config: QuantizedLMConfig):
        super().__init__()
        self.attention = QuantizableWormholeAttention(config)
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),  # Changed from GELU for better quantization
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.layer_norm(intermediate_output + attention_output)
        return layer_output, attention_probs

class QuantizedQuantumLM(nn.Module):
    """Quantized version of the quantum language model"""
    def __init__(self, config: QuantizedLMConfig):
        super().__init__()
        self.config = config

        # Embeddings (not quantized)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer layers (will be quantized)
        self.layers = nn.ModuleList([
            QuantizableTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])

        # Language modeling head (will be quantized)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        self.tie_weights()

        # Quantization-specific attributes
        self.is_quantized = False
        self.quantization_dtype = config.quantization_dtype
        self.quantization_scheme = config.quantization_scheme

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def tie_weights(self):
        self.lm_head.weight = self.word_embeddings.weight

    def quantize(self):
        """Apply quantization to the model"""
        if self.is_quantized:
            return

        if self.quantization_scheme == 'dynamic':
            # Dynamic quantization
            self.layers = torch.quantization.quantize_dynamic(
                self.layers,
                {nn.Linear},
                dtype=self.quantization_dtype
            )
            self.lm_head = torch.quantization.quantize_dynamic(
                self.lm_head,
                {nn.Linear},
                dtype=self.quantization_dtype
            )
        else:
            # Static quantization (requires calibration)
            self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self, inplace=True)
            # Calibration would be done here
            torch.quantization.convert(self, inplace=True)

        self.is_quantized = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        batch_size, seq_length = input_ids.size()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Generate embeddings (not quantized)
        embeddings = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        hidden_states = self.layer_norm(embeddings)
        hidden_states = self.dropout(hidden_states)

        attention_weights = []
        
        # Pass through quantized transformer layers
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            attention_weights.append(attention_probs)

        # Get logits through quantized head
        lm_logits = self.lm_head(hidden_states)

        outputs = {
            'logits': lm_logits,
            'hidden_states': hidden_states,
            'attention_weights': attention_weights
        }

        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs['loss'] = loss
            outputs['perplexity'] = torch.exp(loss)

        return outputs

def save_quantized_model(model: QuantizedQuantumLM, path: str):
    """Save quantized model with metadata"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True) # If dirname is empty (no dir specified), default to current dir
 
    
    # Save quantization info
    metadata = {
        'is_quantized': model.is_quantized,
        'quantization_dtype': str(model.quantization_dtype),
        'quantization_scheme': model.quantization_scheme,
        'config': vars(model.config)
    }
    
    # Save model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, path)

def load_quantized_model(path: str) -> QuantizedQuantumLM:
    """Load quantized model with metadata"""
    checkpoint = torch.load(path)
    
    # Load metadata
    metadata = checkpoint['metadata']
    config = QuantizedLMConfig(**metadata['config'])
    
    # Initialize model
    model = QuantizedQuantumLM(config)
    
    # Quantize if necessary
    if metadata['is_quantized']:
        model.quantize()
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = QuantizedLMConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=6,
        vocab_size=30522,
        wormhole_hidden_size=64,
        batch_size=8,
        num_epochs=3,
        quantization_dtype=torch.qint8,
        quantization_scheme='dynamic'
    )
    
    # Initialize model
    model = QuantizedQuantumLM(config)
    
    # Train model (simplified example)
    # ... training code here ...
    
    # Quantize model
    model.quantize()
    
    # Save quantized model
    save_quantized_model(model, 'quantized_model.pth')
    
    # Load quantized model
    loaded_model = load_quantized_model('quantized_model.pth')
    
    print("Model successfully quantized, saved, and loaded!")
