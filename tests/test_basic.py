"""
Basic tests for Fatigue Detection System
Tests repository structure and basic functionality
"""

import os
import sys


class TestRepositoryStructure:
    """Test if required directories and files exist"""
    
    def test_readme_exists(self):
        """Check if README.md exists"""
        assert os.path.exists('README.md'), "README.md file should exist"
    
    def test_src_directory_exists(self):
        """Check if Src/code directory exists"""
        assert os.path.exists('Src'), "Src directory should exist"
        assert os.path.exists('Src/code'), "Src/code directory should exist"
    
    def test_doc_directory_exists(self):
        """Check if DOC directory exists"""
        assert os.path.exists('DOC'), "DOC directory should exist"
    
    def test_data_directory_exists(self):
        """Check if Data directory exists"""
        assert os.path.exists('Data'), "Data directory should exist"


class TestReadme:
    """Test README.md content"""
    
    def test_readme_not_empty(self):
        """Check if README.md has content"""
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 100, "README.md should have meaningful content"
    
    def test_readme_mentions_fatigue(self):
        """Check if README mentions fatigue detection"""
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read().lower()
            assert 'fatigue' in content, "README should mention fatigue detection"


class TestPythonEnvironment:
    """Test Python environment and basic imports"""
    
    def test_python_version(self):
        """Check Python version is 3.7+"""
        assert sys.version_info >= (3, 7), "Python version should be 3.7 or higher"
    
    def test_can_import_os(self):
        """Test basic Python imports work"""
        import os
        assert os is not None
    
    def test_can_import_sys(self):
        """Test sys module import"""
        import sys
        assert sys is not None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
