#!/usr/bin/env python3
"""
Test script for the new utility modules.
"""
import os
import tempfile
import shutil

import torch
import numpy as np
import pandas as pd
from gymnasium import spaces

from utils.device_utils import get_device, get_device_or_fallback
from utils.observation_utils import (
    process_observation_to_tensor, 
    calculate_observation_size,
    get_observation_keys
)
from utils.timestamp_utils import (
    convert_to_datetime,
    calculate_time_difference_minutes,
    process_timestamp_columns
)
from utils.file_utils import (
    ensure_directory_exists,
    save_numpy_csv,
    save_text_file,
    create_timestamped_directory
)

def test_device_utils():
    """Test device utilities."""
    print("Testing device utilities...")
    
    device = get_device()
    print(f"Auto-detected device: {device}")
    
    # Test fallback
    preferred_device = torch.device("cpu")
    fallback_device = get_device_or_fallback(preferred_device)
    assert fallback_device == preferred_device
    
    auto_device = get_device_or_fallback(None)
    assert auto_device == device
    
    print("✅ Device utilities test passed!")

def test_observation_utils():
    """Test observation utilities."""
    print("Testing observation utilities...")
    
    # Create mock observation space
    obs_space = {
        "pos": spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32),
        "inventory": spaces.Discrete(5),
    }
    
    # Test observation size calculation
    size = calculate_observation_size(obs_space)
    assert size == 3  # 2 for pos + 1 for inventory
    
    # Test observation keys
    keys = get_observation_keys(obs_space)
    assert keys == ["pos", "inventory"]
    
    # Test observation processing
    obs_dict = {
        "pos": np.array([1.0, 2.0], dtype=np.float32),
        "inventory": 3
    }
    
    device = torch.device("cpu")
    tensor_parts = process_observation_to_tensor(obs_dict, keys, device)
    assert len(tensor_parts) == 2
    
    print("✅ Observation utilities test passed!")

def test_timestamp_utils():
    """Test timestamp utilities."""
    print("Testing timestamp utilities...")
    
    # Test datetime conversion
    timestamps = ["2023-01-01 12:00:00", "2023-01-01 13:00:00"]
    converted = convert_to_datetime(timestamps)
    assert len(converted) == 2
    
    # Test time difference calculation
    start_times = pd.to_datetime(["2023-01-01 12:00:00", "2023-01-01 14:00:00"])
    end_times = pd.to_datetime(["2023-01-01 13:00:00", "2023-01-01 15:00:00"])
    
    diff_minutes = calculate_time_difference_minutes(start_times, end_times)
    assert diff_minutes[0] == 60.0  # 1 hour = 60 minutes
    assert diff_minutes[1] == 60.0
    
    print("✅ Timestamp utilities test passed!")

def test_file_utils():
    """Test file utilities."""
    print("Testing file utilities...")
    
    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test directory creation
        sub_dir = os.path.join(test_dir, "subdir")
        ensure_directory_exists(sub_dir)
        assert os.path.exists(sub_dir)
        
        # Test file saving
        test_data = np.array([[1, 2], [3, 4]])
        csv_path = os.path.join(test_dir, "test.csv")
        save_numpy_csv(csv_path, test_data, delimiter=";")
        assert os.path.exists(csv_path)
        
        # Test text file saving
        text_path = os.path.join(test_dir, "test.txt")
        save_text_file(text_path, "Hello, World!")
        assert os.path.exists(text_path)
        
        # Test timestamped directory creation
        timestamped_dir = create_timestamped_directory(test_dir, "experiment")
        assert os.path.exists(timestamped_dir)
        assert "experiment_" in os.path.basename(timestamped_dir)
        
        print("✅ File utilities test passed!")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)

def main():
    """Run all utility tests."""
    print("=== UTILITY MODULES TEST ===\n")
    
    test_device_utils()
    test_observation_utils()
    test_timestamp_utils()
    test_file_utils()
    
    print("\n✅ All utility modules tests passed!")
    print("Refactoring successfully consolidated duplicated code!")

if __name__ == "__main__":
    main()