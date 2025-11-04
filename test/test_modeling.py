import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from modeling import split_data, train_model, add_predictions, FEATURES, TARGET


@pytest.fixture
def sample_data():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "age": np.random.randint(18, 90, n_samples),
        "height": np.random.uniform(150, 200, n_samples),
        "weight": np.random.uniform(50, 120, n_samples),
        "aids": np.random.randint(0, 2, n_samples),
        "cirrhosis": np.random.randint(0, 2, n_samples),
        "hepatic_failure": np.random.randint(0, 2, n_samples),
        "immunosuppression": np.random.randint(0, 2, n_samples),
        "leukemia": np.random.randint(0, 2, n_samples),
        "lymphoma": np.random.randint(0, 2, n_samples),
        "solid_tumor_with_metastasis": np.random.randint(0, 2, n_samples),
        "diabetes_mellitus": np.random.randint(0, 2, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    train_df, _ = split_data(sample_data)
    return train_model(train_df)


class TestSplitData:
    """Tests for split_data function."""
    
    def test_split_data_returns_two_dataframes(self, sample_data):
        """Test that split_data returns two dataframes."""
        train_df, test_df = split_data(sample_data)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
    
    def test_split_data_correct_sizes(self, sample_data):
        """Test that split creates correct train/test sizes."""
        train_df, test_df = split_data(sample_data, test_size=0.2)
        assert len(train_df) == 80
        assert len(test_df) == 20
    
    def test_split_data_custom_test_size(self, sample_data):
        """Test split with custom test_size."""
        train_df, test_df = split_data(sample_data, test_size=0.3)
        assert len(train_df) == 70
        assert len(test_df) == 30
    
    def test_split_data_reproducibility(self, sample_data):
        """Test that split is reproducible with same random_state."""
        train_df1, test_df1 = split_data(sample_data, random_state=42)
        train_df2, test_df2 = split_data(sample_data, random_state=42)
        pd.testing.assert_frame_equal(train_df1, train_df2)
        pd.testing.assert_frame_equal(test_df1, test_df2)
    
    def test_split_data_no_overlap(self, sample_data):
        """Test that train and test sets don't overlap."""
        train_df, test_df = split_data(sample_data)
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        assert len(train_indices.intersection(test_indices)) == 0


class TestTrainModel:
    """Tests for train_model function."""
    
    def test_train_model_returns_logistic_regression(self, sample_data):
        """Test that train_model returns LogisticRegression instance."""
        train_df, _ = split_data(sample_data)
        model = train_model(train_df)
        assert isinstance(model, LogisticRegression)
    
    def test_train_model_is_fitted(self, sample_data):
        """Test that returned model is fitted."""
        train_df, _ = split_data(sample_data)
        model = train_model(train_df)
        # Check if model has been fitted by checking for coef_ attribute
        assert hasattr(model, "coef_")
        assert model.coef_.shape[1] == len(FEATURES)
    
    def test_train_model_with_default_features(self, sample_data):
        """Test training with default FEATURES."""
        train_df, _ = split_data(sample_data)
        model = train_model(train_df)
        assert model.coef_.shape[1] == len(FEATURES)
    
    def test_train_model_with_custom_features(self, sample_data):
        """Test training with custom feature list."""
        train_df, _ = split_data(sample_data)
        custom_features = ["age", "height", "weight"]
        model = train_model(train_df, feature_list=custom_features)
        assert model.coef_.shape[1] == len(custom_features)
    
    def test_train_model_can_predict(self, sample_data):
        """Test that trained model can make predictions."""
        train_df, test_df = split_data(sample_data)
        model = train_model(train_df)
        predictions = model.predict(test_df[FEATURES])
        assert len(predictions) == len(test_df)
        assert all(pred in [0, 1] for pred in predictions)


class TestAddPredictions:
    """Tests for add_predictions function."""
    
    def test_add_predictions_adds_column(self, sample_data, trained_model):
        """Test that add_predictions adds predictions column."""
        result_df = add_predictions(sample_data.copy(), trained_model)
        assert "predictions" in result_df.columns
    
    def test_add_predictions_correct_shape(self, sample_data, trained_model):
        """Test that predictions have correct shape."""
        result_df = add_predictions(sample_data.copy(), trained_model)
        assert len(result_df["predictions"]) == len(sample_data)
    
    def test_add_predictions_probabilities_range(self, sample_data, trained_model):
        """Test that predictions are probabilities between 0 and 1."""
        result_df = add_predictions(sample_data.copy(), trained_model)
        assert all(0 <= pred <= 1 for pred in result_df["predictions"])
    
    def test_add_predictions_returns_dataframe(self, sample_data, trained_model):
        """Test that add_predictions returns a dataframe."""
        result_df = add_predictions(sample_data.copy(), trained_model)
        assert isinstance(result_df, pd.DataFrame)
    
    def test_add_predictions_with_custom_features(self, sample_data):
        """Test predictions with custom feature list."""
        train_df, test_df = split_data(sample_data)
        custom_features = ["age", "height", "weight"]
        model = train_model(train_df, feature_list=custom_features)
        result_df = add_predictions(test_df.copy(), model, feature_list=custom_features)
        assert "predictions" in result_df.columns
        assert len(result_df["predictions"]) == len(test_df)
    
    def test_add_predictions_does_not_modify_original(self, sample_data, trained_model):
        """Test that original dataframe is not modified if copy is passed."""
        original_columns = sample_data.columns.tolist()
        _ = add_predictions(sample_data.copy(), trained_model)
        assert sample_data.columns.tolist() == original_columns


class TestEndToEndWorkflow:
    """Integration tests for complete modeling workflow."""
    
    def test_complete_workflow(self, sample_data):
        """Test complete train and predict workflow."""
        # Split data
        train_df, test_df = split_data(sample_data)
        
        # Train model
        model = train_model(train_df)
        
        # Add predictions
        train_with_preds = add_predictions(train_df.copy(), model)
        test_with_preds = add_predictions(test_df.copy(), model)
        
        # Verify predictions exist
        assert "predictions" in train_with_preds.columns
        assert "predictions" in test_with_preds.columns
        
        # Verify all predictions are probabilities
        assert all(0 <= pred <= 1 for pred in train_with_preds["predictions"])
        assert all(0 <= pred <= 1 for pred in test_with_preds["predictions"])
    
    def test_workflow_with_custom_features(self, sample_data):
        """Test workflow with custom feature list."""
        custom_features = ["age", "weight", "aids"]
        
        train_df, test_df = split_data(sample_data)
        model = train_model(train_df, feature_list=custom_features)
        test_with_preds = add_predictions(test_df.copy(), model, feature_list=custom_features)
        
        assert "predictions" in test_with_preds.columns
        assert all(0 <= pred <= 1 for pred in test_with_preds["predictions"])