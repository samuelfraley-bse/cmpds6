import pytest
import pandas as pd
import numpy as np
from diabetes_library.preprocessing import drop_invalid_rows, fill_missing_values, encode_gender, encode_ethnicity


class TestDropInvalidRows:
    """Tests for the drop_invalid_rows function"""
    
    def test_keeps_valid_rows(self):
        """Test for required columns are present"""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "ethnicity": ["Asian", "White", "Hispanic"]
        })
        result = drop_invalid_rows(df)
        assert len(result) == 3
    
    def test_drops_row_with_nan_age(self):
        """Test that rows with NaN in age are dropped"""
        df = pd.DataFrame({
            "age": [25, np.nan, 35],
            "gender": ["M", "F", "M"],
            "ethnicity": ["Asian", "White", "Hispanic"]
        })
        result = drop_invalid_rows(df)
        assert len(result) == 2
        assert result["age"].tolist() == [25, 35]
    
    def test_drops_row_with_nan_gender(self):
        """Test that rows with NaN in gender are dropped"""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "gender": ["M", np.nan, "M"],
            "ethnicity": ["Asian", "White", "Hispanic"]
        })
        result = drop_invalid_rows(df)
        assert len(result) == 2
    
    def test_drops_row_with_nan_ethnicity(self):
        """Test that rows with NaN in ethnicity are dropped"""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "ethnicity": ["Asian", np.nan, "Hispanic"]
        })
        result = drop_invalid_rows(df)
        assert len(result) == 2
    
    def test_keeps_rows_with_nan_in_other_columns(self):
        """Test that rows with NaN in non-required columns are kept"""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "ethnicity": ["Asian", "White", "Hispanic"],
            "height": [170, np.nan, 180]
        })
        result = drop_invalid_rows(df)
        assert len(result) == 3
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        df = pd.DataFrame({
            "age": [],
            "gender": [],
            "ethnicity": []
        })
        result = drop_invalid_rows(df)
        assert len(result) == 0
    
    def test_all_rows_invalid(self):
        """Test when all rows have NaN in required columns"""
        df = pd.DataFrame({
            "age": [np.nan, np.nan],
            "gender": ["M", "F"],
            "ethnicity": ["Asian", "White"]
        })
        result = drop_invalid_rows(df)
        assert len(result) == 0


class TestFillMissingValues:
    """Tests for fill_missing_values function"""
    
    def test_fills_missing_height_with_mean(self):
        """Test that missing height values are imputed with mean"""
        df = pd.DataFrame({
            "height": [170, np.nan, 180, 190]
        })
        result = fill_missing_values(df)
        expected_mean = (170 + 180 + 190) / 3
        assert result["height"].iloc[1] == expected_mean
    
    def test_fills_missing_weight_with_mean(self):
        """Test that missing weight values are imputed with mean"""
        df = pd.DataFrame({
            "weight": [60, 70, np.nan, 80]
        })
        result = fill_missing_values(df)
        expected_mean = (60 + 70 + 80) / 3
        assert result["weight"].iloc[2] == expected_mean
    
    def test_preserves_existing_values(self):
        """Test that existing values are not changed"""
        df = pd.DataFrame({
            "height": [170, 175, 180],
            "weight": [60, 70, 80]
        })
        result = fill_missing_values(df)
        assert result["height"].tolist() == [170, 175, 180]
        assert result["weight"].tolist() == [60, 70, 80]
    
    def test_handles_missing_columns(self):
        """Test that function works even when height/weight columns don't exist"""
        df = pd.DataFrame({
            "age": [25, 30, 35]
        })
        result = fill_missing_values(df)
        assert "height" not in result.columns
        assert "weight" not in result.columns
    
    def test_handles_only_height_column(self):
        """Test with only height column present"""
        df = pd.DataFrame({
            "height": [170, np.nan, 180]
        })
        result = fill_missing_values(df)
        assert result["height"].iloc[1] == 175
    
    def test_handles_only_weight_column(self):
        """Test with only weight column present"""
        df = pd.DataFrame({
            "weight": [60, np.nan, 80]
        })
        result = fill_missing_values(df)
        assert result["weight"].iloc[1] == 70
    
    def test_preserves_other_columns(self):
        """Test that other columns are not affected"""
        df = pd.DataFrame({
            "height": [170, np.nan, 180],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"]
        })
        result = fill_missing_values(df)
        assert result["age"].tolist() == [25, 30, 35]
        assert result["gender"].tolist() == ["M", "F", "M"]


class TestEncodeGender:
    """Tests for encode_gender function."""
    
    def test_encodes_male_as_one(self):
        """Test that M is encoded as 1"""
        df = pd.DataFrame({
            "gender": ["M", "M"]
        })
        result = encode_gender(df)
        assert result["gender"].tolist() == [1, 1]
    
    def test_encodes_female_as_zero(self):
        """Test that F is encoded as 0"""
        df = pd.DataFrame({
            "gender": ["F", "F"]
        })
        result = encode_gender(df)
        assert result["gender"].tolist() == [0, 0]
    
    def test_encodes_mixed_genders(self):
        """Test encoding with both M and F"""
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F"]
        })
        result = encode_gender(df)
        assert result["gender"].tolist() == [1, 0, 1, 0]
    
    def test_handles_invalid_values(self):
        """Test that invalid gender values become NaN"""
        df = pd.DataFrame({
            "gender": ["M", "F", "X", "Other"]
        })
        result = encode_gender(df)
        assert result["gender"].iloc[0] == 1
        assert result["gender"].iloc[1] == 0
        assert pd.isna(result["gender"].iloc[2])
        assert pd.isna(result["gender"].iloc[3])
    
    def test_preserves_other_columns(self):
        """Test that other columns are not affected"""
        df = pd.DataFrame({
            "gender": ["M", "F"],
            "age": [25, 30],
            "height": [170, 165]
        })
        result = encode_gender(df)
        assert result["age"].tolist() == [25, 30]
        assert result["height"].tolist() == [170, 165]


class TestEncodeEthnicity:
    """Tests for encode_ethnicity function."""
    
    def test_creates_dummy_columns(self):
        """Test that dummy columns are created for ethnicity values"""
        df = pd.DataFrame({
            "ethnicity": ["Asian", "White", "Hispanic"]
        })
        result = encode_ethnicity(df)
        assert "ethnicity" not in result.columns
        # Should have n-1 columns (drop_first=True)
        ethnicity_cols = [col for col in result.columns if col.startswith("ethnicity_")]
        assert len(ethnicity_cols) == 2
    
    def test_drops_first_category(self):
        """Test that first category is dropped (drop_first=True)"""
        df = pd.DataFrame({
            "ethnicity": ["Asian", "White", "Hispanic"]
        })
        result = encode_ethnicity(df)
        ethnicity_cols = sorted([col for col in result.columns if col.startswith("ethnicity_")])
        # First category (Asian alphabetically) should not have a column
        assert "ethnicity_Asian" not in result.columns
    
    def test_preserves_other_columns(self):
        """Test that other columns are preserved"""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "gender": [1, 0, 1],
            "ethnicity": ["Asian", "White", "Hispanic"]
        })
        result = encode_ethnicity(df)
        assert "age" in result.columns
        assert "gender" in result.columns
        assert result["age"].tolist() == [25, 30, 35]
        assert result["gender"].tolist() == [1, 0, 1]
    
    def test_correct_encoding_values(self):
        """Test that dummy encoding produces correct binary values"""
        df = pd.DataFrame({
            "ethnicity": ["White", "White", "Hispanic"]
        })
        result = encode_ethnicity(df)
        # All values in dummy columns should be 0 or 1
        for col in result.columns:
            if col.startswith("ethnicity_"):
                assert result[col].isin([0, 1]).all()
    
    def test_handles_single_ethnicity(self):
        """Test with only one ethnicity value"""
        df = pd.DataFrame({
            "ethnicity": ["Asian", "Asian", "Asian"]
        })
        result = encode_ethnicity(df)
        # With drop_first=True and only one category, no ethnicity columns should remain
        ethnicity_cols = [col for col in result.columns if col.startswith("ethnicity_")]
        assert len(ethnicity_cols) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])