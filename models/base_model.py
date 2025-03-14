from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class StateError(ModelError):
    """Exception raised for invalid model state."""
    pass

def validate_state_consistency(state, config):
    """
    Utility function to validate state consistency across models.
    
    Args:
        state: The model state to validate
        config: The configuration to check against
        
    Returns:
        bool: True if state is consistent, False otherwise
    """
    if not state:
        return False
    
    # Check if the state has basic required fields
    required_fields = ['timestamp']
    for field in required_fields:
        if field not in state:
            return False
    
    return True

class ParameterError(ModelError):
    """Exception raised for invalid parameters."""
    pass

class BaseModel(ABC):
    """Base class for all simulation models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            config: Dictionary containing model configuration parameters
            
        Raises:
            ParameterError: If required configuration parameters are missing
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state: Dict[str, Any] = {}
        self.shared_state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_update = None
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate model configuration.
        
        Raises:
            ParameterError: If required parameters are missing or invalid
        """
        try:
            required_params = self.get_required_parameters()
            missing_params = [param for param in required_params if param not in self.config]
            
            if missing_params:
                raise ParameterError(f"Missing required parameters: {missing_params}")
        except NotImplementedError:
            # If get_required_parameters is not implemented, assume no required parameters
            pass
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required configuration parameters.
        
        Returns:
            List[str]: List of required parameter names
        """
        return []
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize model state and parameters.
        
        Raises:
            StateError: If initialization fails
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """
        Execute one simulation step.
        
        Raises:
            StateError: If step execution fails
        """
        pass
    
    @abstractmethod
    def update(self) -> None:
        """
        Update model state based on step results.
        
        Raises:
            StateError: If state update fails
        """
        # Ensure timestamp is set
        self.state['timestamp'] = datetime.now().isoformat()
    
    def update_shared_state(self, shared_state: Dict[str, Dict[str, Any]]) -> None:
        """
        Update shared state from other models.
        
        Args:
            shared_state: Dictionary containing state from other models
        """
        self.shared_state = shared_state
    
    def save_state(self) -> None:
        """Save current state to history."""
        try:
            state_copy = self.state.copy()
            state_copy['timestamp'] = datetime.now().isoformat()
            if validate_state_consistency(state_copy, self.config):
                self.history.append(state_copy)
                self.last_update = datetime.now()
            else:
                self.logger.warning("State validation failed, state not saved to history")
                raise StateError("State validation failed, state not saved to history")
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            raise StateError(f"State save failed: {str(e)}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current model state.
        
        Returns:
            Dict[str, Any]: Current model state
        """
        return self.state.copy()
    
    def get_shared_state(self) -> Dict[str, Any]:
        """
        Get current shared state.
        
        Returns:
            Dict[str, Any]: Current shared state
        """
        return self.shared_state.copy()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get simulation history.
        
        Returns:
            List[Dict[str, Any]]: List of historical states
        """
        return self.history.copy()
    
    def validate_state(self) -> bool:
        """
        Validate current model state.
        
        Returns:
            bool: True if state is valid, False otherwise
            
        Raises:
            StateError: If state validation fails
        """
        try:
            # Check that state is not empty
            if not self.state:
                self.logger.warning("State is empty")
                return False
                
            # Ensure basic fields are present
            required_fields = ['timestamp']
            for field in required_fields:
                if field not in self.state:
                    self.state[field] = datetime.now().isoformat()
            
            # Allow model-specific validation
            return self._validate_model_state()
        except Exception as e:
            self.logger.error(f"State validation failed: {str(e)}")
            raise StateError(f"State validation failed: {str(e)}")
    
    def _validate_model_state(self) -> bool:
        """
        Model-specific state validation. Override in subclasses for custom validation.
        
        Returns:
            bool: True if model-specific state is valid
        """
        # Default implementation returns True
        return True
    
    def reset(self) -> None:
        """
        Reset model to initial state.
        
        Raises:
            StateError: If reset fails
        """
        try:
            self.state = {}
            self.shared_state = {}
            self.history = []
            self.initialize()
        except Exception as e:
            self.logger.error(f"Reset failed: {str(e)}")
            raise StateError(f"Reset failed: {str(e)}")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current model metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        return {}
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a model parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Raises:
            ParameterError: If parameter is invalid or setting fails
        """
        try:
            if name in self.config:
                self.config[name] = value
            else:
                self.logger.warning(f"Parameter {name} not found in config")
                raise ParameterError(f"Parameter {name} not found in config")
        except Exception as e:
            self.logger.error(f"Failed to set parameter {name}: {str(e)}")
            raise ParameterError(f"Parameter setting failed: {str(e)}")
    
    def get_parameter(self, name: str) -> Any:
        """
        Get a model parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Any: Parameter value
            
        Raises:
            ParameterError: If parameter is not found
        """
        if name in self.config:
            return self.config[name]
        else:
            raise ParameterError(f"Parameter {name} not found in config")
    
    def log_state(self) -> None:
        """Log current model state."""
        self.logger.info(f"Current state: {self.state}")
    
    def log_metrics(self) -> None:
        """Log current model metrics."""
        metrics = self.get_metrics()
        self.logger.info(f"Current metrics: {metrics}")
    
    def get_last_update_time(self) -> datetime:
        """
        Get timestamp of last state update.
        
        Returns:
            datetime: Timestamp of last update
        """
        return self.last_update