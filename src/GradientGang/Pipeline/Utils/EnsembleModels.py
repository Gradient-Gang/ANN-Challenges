import torch


class EnsembleModel(torch.nn.Module):
    """
    Ensemble wrapper for multiple models with averaged predictions.
    
    This class combines multiple trained models (Direct or Autoencoder architectures)
    into a single ensemble that averages their predictions. Compatible with
    SubmissionGenerator for generating test predictions.
    
    Supports:
    - Direct classification models
    - Autoencoder-based models
    - Mixed ensembles of different architectures
    - Automatic device management
    - Soft voting (average logits before argmax)
    
    Example:
        >>> models = [model1, model2, model3]
        >>> model_types = ["Direct", "Autoencoder", "Direct"]
        >>> ensemble = EnsembleModel(models, model_types)
        >>> predictions = ensemble((time_series, global_features))
    """
    
    def __init__(self, models, model_types):
        """
        Initialize ensemble model with list of trained models.
        
        Args:
            models (list): List of trained PyTorch models to ensemble.
                          All models should be on the same device.
            model_types (list): List of strings indicating model type for each model.
                               Options: 'Direct' or 'Autoencoder'.
                               Must have same length as models list.
        
        Raises:
            ValueError: If models list is empty or model_types length doesn't match
        """
        super().__init__()
        
        if not models:
            raise ValueError("Models list cannot be empty")
        
        if len(models) != len(model_types):
            raise ValueError(f"Length mismatch: {len(models)} models but {len(model_types)} types")
        
        self.models = torch.nn.ModuleList(models)
        self.model_types = model_types
        
        # Get device from first model's parameters
        try:
            self.device = next(models[0].parameters()).device
        except StopIteration:
            # Model has no parameters, default to CPU
            self.device = torch.device('cpu')
    
    def forward(self, x):
        """
        Forward pass through all models with averaged predictions.
        
        Performs inference through each model in the ensemble and averages
        their logits (soft voting). This typically provides better performance
        than hard voting (averaging predicted classes).
        
        Args:
            x: Input features. Can be:
               - Tuple of (timeSeries, globalFeats) for models with dual inputs
               - Single tensor for models with single input
        
        Returns:
            torch.Tensor: Averaged logits from all models (batch_size, num_classes)
        
        Note:
            Models are automatically set to eval mode during forward pass.
            No gradients are computed to save memory.
        """
        all_logits = []
        
        # Parse input format: tuple (timeSeries, globalFeats) or single tensor
        if isinstance(x, tuple) and len(x) == 2:
            timeSeries, globalFeats = x
        else:
            # Single input - wrap as tuple for consistency
            timeSeries = x
            globalFeats = None
        
        # Collect predictions from all models
        for i, model in enumerate(self.models):
            # Ensure model is in eval mode
            model.eval()
            
            with torch.no_grad():
                model_type = self.model_types[i]
                
                # Call model with appropriate input format based on architecture type
                if model_type == "Direct":
                    # Direct models: forward(timeSeries, globalFeats)
                    output = model(timeSeries, globalFeats)
                elif model_type == "Autoencoder":
                    # Autoencoder models: forward((timeSeries, globalFeats))
                    # Returns tuple: (classification_logits, reconstructions)
                    output = model((timeSeries, globalFeats))
                else:
                    raise ValueError(f"Unknown model type: {model_type}. Expected 'Direct' or 'Autoencoder'")
                
                # Extract logits from output (handle tuple outputs from autoencoders)
                if isinstance(output, tuple):
                    # Autoencoder returns (logits, reconstructions) - take logits
                    logits = output[0]
                else:
                    # Direct model returns logits directly
                    logits = output
                
                all_logits.append(logits)
        
        # Average logits across all models (soft voting)
        # Shape: (num_models, batch_size, num_classes) -> (batch_size, num_classes)
        ensemble_logits = torch.mean(torch.stack(all_logits), dim=0)
        
        return ensemble_logits
    
    def eval(self):
        """
        Set all models in ensemble to evaluation mode.
        
        Returns:
            self: Returns self for method chaining
        """
        for model in self.models:
            model.eval()
        return self
    
    def train(self, mode=True):
        """
        Set all models in ensemble to training mode.
        
        Note: Typically ensembles are used for inference only,
        but this method is provided for completeness.
        
        Args:
            mode (bool): If True, set to training mode. If False, set to eval mode.
        
        Returns:
            self: Returns self for method chaining
        """
        for model in self.models:
            model.train(mode)
        return self
    
    def to(self, device):
        """
        Move all models in ensemble to specified device.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0'))
        
        Returns:
            self: Returns self for method chaining
        """
        super().to(device)
        self.device = device
        return self
    
    def __len__(self):
        """Return number of models in ensemble."""
        return len(self.models)
    
    def __repr__(self):
        """String representation of ensemble."""
        model_info = ", ".join([f"{t}({type(m).__name__})" for t, m in zip(self.model_types, self.models)])
        return f"EnsembleModel(n_models={len(self)}, models=[{model_info}], device={self.device})"
