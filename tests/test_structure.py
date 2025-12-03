import pytest
import os

def test_project_directories_exist():
    """Test if main project directories exist"""
    required_dirs = ['DOC', 'Data', 'Results', 'Src']
    
    for dir_name in required_dirs:
        assert os.path.exists(dir_name), f"{dir_name} directory should exist"

def test_readme_exists():
    """Test if README.md exists"""
    assert os.path.exists('README.md'), "README.md should exist"

def test_src_structure():
    """Test Src directory structure"""
    assert os.path.exists('Src/code'), "Src/code directory should exist"
    assert os.path.exists('Src/models'), "Src/models directory should exist"

def test_code_files_exist():
    """Test if main Python files exist"""
    code_files = [
        'Src/code/blink_detector.py',
        'Src/code/camera_test.py',
        'Src/code/fatigue_detector.py',
        'Src/code/train_mouth.py'
    ]
    
    for file_path in code_files:
        assert os.path.exists(file_path), f"{file_path} should exist"

def test_data_directory():
    """Test Data directory exists"""
    assert os.path.exists('Data'), "Data directory should exist"

def test_documentation_exists():
    """Test if documentation files exist"""
    assert os.path.exists('DOC'), "DOC directory should exist"
