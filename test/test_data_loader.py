import pytest


class TestImports:
    """Test suite to verify all hw5lib modules can be imported successfully."""
    
    def test_data_module_import(self):
        """Test that data.py module imports successfully."""
        from hw5lib.data import DataLoader
        assert DataLoader is not None, "DataLoader class should be importable"
    
    def test_preprocess_module_import(self):
        """Test that preprocess.py module imports successfully."""
        from hw5lib.preprocess import NaNRowRemover
        assert NaNRowRemover is not None, "NaNRowRemover class should be importable"
    
    def test_features_module_import(self):
        """Test that features.py module imports successfully."""
        from hw5lib.features import BMICalculator
        assert BMICalculator is not None, "BMICalculator class should be importable"
    
    def test_model_module_import(self):
        """Test that model.py module imports successfully."""
        from hw5lib.model import DiabetesModel
        assert DiabetesModel is not None, "DiabetesModel class should be importable"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])