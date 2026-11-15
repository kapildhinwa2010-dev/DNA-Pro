"""
Tests for DNA-Pro module
"""

import pytest
import torch
import torch.nn as nn
from dna_pro import DNAPro, DNAProWrapper, DNAProConfig, create_dna_pro


class TestDNAProInitialization:
    """Test DNAPro initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        dna = DNAPro()
        assert dna.num_freqs == 64
        assert dna.hidden_size == 128
        assert dna.chunk_size == 16384
        assert dna.inference_mode == True
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        dna = DNAPro(
            num_freqs=32,
            hidden_size=64,
            chunk_size=8192,
            inference_mode=False
        )
        assert dna.num_freqs == 32
        assert dna.hidden_size == 64
        assert dna.chunk_size == 8192
        assert dna.inference_mode == False
    
    def test_buffer_registration(self):
        """Test that B is registered as buffer"""
        dna = DNAPro()
        assert hasattr(dna, 'B')
        assert isinstance(dna.B, torch.Tensor)
        assert dna.B.shape == (64, 2)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        dna = DNAPro(num_freqs=32, hidden_size=64)
        count = dna.count_parameters()
        # 32*2*64 + 64 + 64*64 + 64 + 64*1 + 1 + 1
        expected = 32*2*64 + 64 + 64*64 + 64 + 64*1 + 1 + 1
        assert count == expected


class TestDNAProWeightGeneration:
    """Test weight generation"""
    
    def test_generate_small_matrix(self):
        """Test generating small weight matrix"""
        dna = DNAPro()
        W = dna.generate_weights(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_large_matrix(self):
        """Test generating large weight matrix"""
        dna = DNAPro(chunk_size=16384)
        W = dna.generate_weights(100, 100)
        assert W.shape == (100, 100)
    
    def test_generate_matrix_device(self):
        """Test weight generation on CPU"""
        dna = DNAPro()
        W = dna.generate_weights(10, 10, device=torch.device('cpu'))
        assert W.device.type == 'cpu'
    
    def test_weight_matrix_properties(self):
        """Test weight matrix statistical properties"""
        dna = DNAPro()
        W = dna.generate_weights(100, 100)
        # Check that weights are not all zeros
        assert W.abs().sum() > 0
        # Check that weights have reasonable scale
        assert W.std() > 0


class TestDNAProChunkedProcessing:
    """Test chunked processing"""
    
    def test_chunked_vs_unchunked(self):
        """Test that chunked and unchunked produce consistent results"""
        dna1 = DNAPro(chunk_size=10000)
        dna2 = DNAPro(chunk_size=100)
        
        # Copy parameters from dna1 to dna2
        dna2.net.load_state_dict(dna1.net.state_dict())
        dna2.B.data = dna1.B.data.clone()
        dna2.scale.data = dna1.scale.data.clone()
        
        W1 = dna1.generate_weights(100, 100)
        W2 = dna2.generate_weights(100, 100)
        
        # Should be close (may differ slightly due to chunking)
        assert torch.allclose(W1, W2, rtol=1e-4, atol=1e-4)
    
    def test_large_matrix_chunking(self):
        """Test that large matrices are processed correctly"""
        dna = DNAPro(chunk_size=100)
        W = dna.generate_weights(500, 500)
        assert W.shape == (500, 500)
        # Check consistency
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


class TestDNAProLayerWeights:
    """Test layer weight generation"""
    
    def test_generate_layer_weights(self):
        """Test generating weights for multiple layers"""
        dna = DNAPro()
        layer_config = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = dna.generate_layer_weights(layer_config)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)


class TestDNAProModelWeights:
    """Test model weight generation"""
    
    def test_generate_model_weights(self):
        """Test generating weights for a model"""
        dna = DNAPro()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        weights = dna.generate_model_weights(model)
        # Should have weights for both linear layers
        assert len(weights) >= 2


class TestDNAProWrapper:
    """Test DNAProWrapper"""
    
    def test_generate_weights(self):
        """Test generating weights through wrapper"""
        wrapper = DNAProWrapper()
        shapes = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = wrapper.generate_weights(shapes)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)
    
    def test_generate_for_model(self):
        """Test generating weights for a model through wrapper"""
        wrapper = DNAProWrapper()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        weights = wrapper.generate_for_model(model)
        assert len(weights) >= 2


class TestCreateDNAPro:
    """Test create_dna_pro factory function"""
    
    def test_default_creation(self):
        """Test default DNA-Pro creation"""
        dna = create_dna_pro()
        assert isinstance(dna, DNAPro)
        assert dna.num_freqs == 64
        assert dna.hidden_size == 128
    
    def test_custom_creation(self):
        """Test custom DNA-Pro creation"""
        dna = create_dna_pro(
            num_freqs=32,
            hidden_size=64,
            chunk_size=8192,
            inference_mode=False
        )
        assert dna.num_freqs == 32
        assert dna.hidden_size == 64


class TestDNAProConfig:
    """Test DNAProConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = DNAProConfig()
        assert config.num_freqs == 64
        assert config.hidden_size == 128
        assert config.chunk_size == 16384
        assert config.inference_mode == True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DNAProConfig(
            num_freqs=32,
            hidden_size=64,
            inference_mode=False
        )
        assert config.num_freqs == 32
        assert config.hidden_size == 64
        assert config.inference_mode == False


class TestInferenceMode:
    """Test inference mode"""
    
    def test_inference_mode_frozen(self):
        """Test that inference mode freezes parameters"""
        dna = DNAPro(inference_mode=True)
        for param in dna.parameters():
            assert not param.requires_grad
    
    def test_training_mode_not_frozen(self):
        """Test that training mode doesn't freeze parameters"""
        dna = DNAPro(inference_mode=False)
        for param in dna.parameters():
            assert param.requires_grad


class TestCompressionRatio:
    """Test compression ratio calculations"""
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        dna = DNAPro(num_freqs=64, hidden_size=128)
        
        # DNA has: 64*2*128 + 128 + 128*128 + 128 + 128*1 + 1 + 1
        dna_params = dna.count_parameters()
        
        # For a 1M parameter model
        original_params = 1_000_000
        ratio = dna.compression_ratio(original_params)
        
        expected_ratio = original_params / dna_params
        assert abs(ratio - expected_ratio) < 0.01


class TestDeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        dna = DNAPro()
        W = dna.generate_weights(10, 10)
        assert W.device == dna.B.device
        assert W.device == dna.scale.device
        for param in dna.net.parameters():
            assert param.device == dna.B.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        dna = DNAPro()
        W = dna.generate_weights(10, 10)
        assert W.dtype == dna.B.dtype
        assert W.dtype == dna.scale.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
