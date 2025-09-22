"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""


import logging
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional, Union, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
import statsmodels.api as sm


# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}


def create_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, LabelEncoder, LabelEncoder]:
    """
    Create features for the Bayesian healthcare access model.

    This function processes raw healthcare facility data to create meaningful features
    for predicting specialized care availability. It handles missing values, encodes
    categorical variables, and standardizes numerical features.

    Args:
        df (pd.DataFrame): Raw healthcare facilities DataFrame containing facility information
            including facility names, types, locations, population data, and ownership details.

    Returns:
        Tuple[pd.DataFrame, pd.Series, StandardScaler, LabelEncoder, LabelEncoder]:
            - X: Feature matrix with processed numerical and categorical variables
            - y: Target variable (1 for specialized care, 0 otherwise)
            - scaler: Fitted StandardScaler for numerical features
            - county_encoder: Fitted LabelEncoder for county categories
            - owner_encoder: Fitted LabelEncoder for ownership categories

    Raises:
        KeyError: If required columns are missing from the input DataFrame
        ValueError: If the DataFrame is empty or contains invalid data types
        Exception: For any unexpected errors during feature processing

    Example:
        >>> X, y, scaler, county_enc, owner_enc = create_features(facilities_df)
        >>> print(f"Created {X.shape[1]} features for {X.shape[0]} facilities")
    """
    logger.info("Starting feature creation process.")

    try:
        if df.empty:
            logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame cannot be empty")

        # Validate required columns
        required_cols = [
            "facility_n",
            "type",
            "county",
            "owner",
            "population_density_no_per_sq_km",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise KeyError(f"Missing required columns: {missing_cols}")

        df_processed = df.copy()
        logger.debug(f"Processing {len(df_processed)} facilities")

        # Create target variable: specialized care (level 4+ facilities)
        def is_specialized(facility_name: str, facility_type: str) -> int:
            """
            Determine if a facility offers specialized care based on name and type.

            Args:
                facility_name (str): Name of the healthcare facility
                facility_type (str): Type/category of the healthcare facility

            Returns:
                int: 1 if specialized care facility, 0 otherwise
            """
            facility_name = str(facility_name).lower()
            facility_type = str(facility_type).lower()

            specialized_keywords = [
                "hospital",
                "referral",
                "medical center",
                "health centre",
            ]

            # Check facility name
            if any(keyword in facility_name for keyword in specialized_keywords):
                return 1

            # Check facility type
            if any(keyword in facility_type for keyword in specialized_keywords):
                return 1

            # Explicit type matching
            if facility_type in ["hospital", "referral hospital", "county hospital"]:
                return 1

            return 0

        logger.debug("Creating specialized care target variable")
        df_processed["specialized_care"] = df_processed.apply(
            lambda x: is_specialized(x["facility_n"], x["type"]), axis=1
        )

        specialized_count = df_processed["specialized_care"].sum()
        logger.info(
            f"Identified {specialized_count} specialized care facilities out of {len(df_processed)}"
        )

        # Define numerical features
        numerical_features = [
            "population_density_no_per_sq_km",
            "population_x",
            "land_area_sq_km",
            "male",
            "female",
            "households",
            "average_household_size",
        ]

        # Filter existing numerical features
        existing_numerical = [
            f for f in numerical_features if f in df_processed.columns
        ]
        logger.debug(
            f"Found {len(existing_numerical)} numerical features: {existing_numerical}"
        )

        # Create urban/rural indicator
        logger.debug("Creating urban/rural classification")
        df_processed["urban_rural"] = df_processed[
            "population_density_no_per_sq_km"
        ].apply(lambda x: 1 if pd.notna(x) and x > 1000 else 0)
        existing_numerical.append("urban_rural")

        # Handle missing values in numerical features
        logger.debug("Handling missing values in numerical features")
        for feature in existing_numerical:
            if feature in df_processed.columns:
                missing_count = df_processed[feature].isna().sum()
                if missing_count > 0:
                    logger.debug(f"Filling {missing_count} missing values in {feature}")
                    df_processed[feature] = df_processed[feature].fillna(
                        df_processed[feature].median()
                    )

        # Encode categorical variables
        logger.debug("Encoding categorical variables")

        # County encoding
        unique_counties = df_processed["county"].nunique()
        logger.debug(f"Encoding {unique_counties} unique counties")
        county_encoder = LabelEncoder()
        df_processed["county_encoded"] = county_encoder.fit_transform(
            df_processed["county"].fillna("Unknown")
        )

        # Owner encoding
        unique_owners = df_processed["owner"].nunique()
        logger.debug(f"Encoding {unique_owners} unique owner types")
        owner_encoder = LabelEncoder()
        df_processed["owner_encoded"] = owner_encoder.fit_transform(
            df_processed["owner"].fillna("Unknown")
        )

        categorical_features = ["county_encoded", "owner_encoded"]

        # Standardize numerical features
        logger.debug("Standardizing numerical features")
        scaler = StandardScaler()
        df_processed[existing_numerical] = scaler.fit_transform(
            df_processed[existing_numerical]
        )

        # Prepare final feature set
        all_features = existing_numerical + categorical_features
        X = df_processed[all_features]
        y = df_processed["specialized_care"]

        # Log feature summary
        target_distribution = y.value_counts().to_dict()
        logger.info(f"Created {len(all_features)} features:")
        logger.info(f"  - Numerical: {existing_numerical}")
        logger.info(f"  - Categorical: {categorical_features}")
        logger.info(f"  - Target distribution: {target_distribution}")

        return X, y, scaler, county_encoder, owner_encoder

    except KeyError as ke:
        logger.error(f"KeyError in feature creation: {ke}")
        raise
    except ValueError as ve:
        logger.error(f"ValueError in feature creation: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature creation: {e}")
        raise


def build_bayesian_logistic_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Any]:
    """
    Build Bayesian logistic regression model using MCMC sampling.

    This function creates a Bayesian logistic regression model that provides
    uncertainty quantification for predictions through posterior sampling.

    Args:
        X (pd.DataFrame): Feature matrix with processed variables
        y (pd.Series): Binary target variable (specialized care indicator)

    Returns:
        Tuple[Any, Any]:
            - model: Fitted statsmodels BayesLogit model
            - result: MCMC sampling results with posterior distributions

    Raises:
        ValueError: If input data is invalid or model convergence fails
        Exception: For any unexpected errors during model fitting

    Example:
        >>> model, result = build_bayesian_logistic_model(X_train, y_train)
        >>> print(f"Model converged with {result.niter} iterations")
    """
    logger.info("Building Bayesian Logistic Regression Model...")

    try:
        if X.empty or y.empty:
            logger.error("Input data is empty")
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            logger.error(
                f"Feature matrix length ({len(X)}) doesn't match target length ({len(y)})"
            )
            raise ValueError("Feature matrix and target must have same length")

        logger.debug(f"Input data shape: X={X.shape}, y={y.shape}")
        logger.debug(f"Target value counts: {y.value_counts().to_dict()}")

        # Add constant for intercept
        logger.debug("Adding constant term for intercept")
        X_with_const = sm.add_constant(X)

        # Fit Bayesian logistic regression with MCMC
        logger.debug("Initializing Bayesian logistic model")
        model = sm.BayesLogit(y, X_with_const)

        logger.info("Starting MCMC sampling (this may take a few minutes)...")
        result = model.fit(method="mcmc", nburn=1000, niter=5000, tune_interval=100)

        logger.info("✓ Bayesian Logistic model built successfully")
        logger.debug(
            f"MCMC completed with {result.niter} iterations, {result.nburn} burn-in"
        )

        return model, result

    except ValueError as ve:
        logger.error(f"ValueError in Bayesian logistic model: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error building Bayesian logistic model: {e}")
        raise


def build_naive_bayes_model(X: np.ndarray, y: np.ndarray) -> GaussianNB:
    """
    Build Gaussian Naive Bayes classifier for comparison.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target variable

    Returns:
        GaussianNB: Fitted Naive Bayes model

    Raises:
        ValueError: If input data is invalid
        Exception: For any unexpected errors during model fitting

    Example:
        >>> nb_model = build_naive_bayes_model(X_train.values, y_train.values)
        >>> accuracy = nb_model.score(X_test.values, y_test.values)
    """
    logger.info("Building Naive Bayes Model...")

    try:
        if len(X) == 0 or len(y) == 0:
            logger.error("Input arrays are empty")
            raise ValueError("Input arrays cannot be empty")

        logger.debug(
            f"Training Naive Bayes with {len(X)} samples, {X.shape[1]} features"
        )

        model = GaussianNB()
        model.fit(X, y)

        # Calculate training accuracy for logging
        train_accuracy = model.score(X, y)
        logger.info(
            f"✓ Naive Bayes model built successfully (Training accuracy: {train_accuracy:.3f})"
        )

        return model

    except ValueError as ve:
        logger.error(f"ValueError in Naive Bayes model: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error building Naive Bayes model: {e}")
        raise


def build_bayesian_ridge_model(X: np.ndarray, y: np.ndarray) -> BayesianRidge:
    """
    Build Bayesian Ridge regression model for probabilistic predictions.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target variable

    Returns:
        BayesianRidge: Fitted Bayesian Ridge model

    Raises:
        ValueError: If input data is invalid
        Exception: For any unexpected errors during model fitting

    Example:
        >>> br_model = build_bayesian_ridge_model(X_train.values, y_train.values)
        >>> predictions, std = br_model.predict(X_test.values, return_std=True)
    """
    logger.info("Building Bayesian Ridge Model...")

    try:
        if len(X) == 0 or len(y) == 0:
            logger.error("Input arrays are empty")
            raise ValueError("Input arrays cannot be empty")

        logger.debug(
            f"Training Bayesian Ridge with {len(X)} samples, {X.shape[1]} features"
        )

        model = BayesianRidge(compute_score=True)
        model.fit(X, y)

        # Log model parameters
        logger.debug(f"Model alpha: {model.alpha_:.4f}, lambda: {model.lambda_:.4f}")
        logger.info("✓ Bayesian Ridge model built successfully")

        return model

    except ValueError as ve:
        logger.error(f"ValueError in Bayesian Ridge model: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error building Bayesian Ridge model: {e}")
        raise


def predict_probability_bayesian_logistic(
    result: Any, X_new: np.ndarray, feature_names: List[str]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Predict probability using Bayesian logistic regression results.

    Args:
        result: MCMC results from statsmodels BayesLogit
        X_new (np.ndarray): New feature values for prediction
        feature_names (List[str]): Names of features for logging

    Returns:
        Dict[str, Union[float, np.ndarray]]: Prediction results including:
            - mean_probability: Average predicted probability
            - std_probability: Standard deviation of predictions
            - credible_interval_95: 95% credible interval
            - all_samples: All MCMC samples for the prediction

    Raises:
        ValueError: If input data is invalid
        Exception: For any unexpected errors during prediction

    Example:
        >>> pred = predict_probability_bayesian_logistic(result, X_new, feature_names)
        >>> print(f"Probability: {pred['mean_probability']:.3f} ± {pred['std_probability']:.3f}")
    """
    logger.debug("Making Bayesian logistic prediction")

    try:
        if len(X_new) == 0:
            logger.error("Empty prediction input")
            raise ValueError("Prediction input cannot be empty")

        # Add constant to X_new
        X_new_with_const = sm.add_constant(X_new, has_constant="add")
        logger.debug(f"Prediction input shape: {X_new_with_const.shape}")

        # Get posterior predictive distribution
        predictions = result.predict(X_new_with_const)

        mean_prob = np.mean(predictions)
        std_prob = np.std(predictions)
        credible_interval = np.percentile(predictions, [2.5, 97.5])

        logger.debug(
            f"Bayesian logistic prediction - Mean: {mean_prob:.3f}, Std: {std_prob:.3f}"
        )

        return {
            "mean_probability": mean_prob,
            "std_probability": std_prob,
            "credible_interval_95": credible_interval,
            "all_samples": predictions,
        }

    except ValueError as ve:
        logger.error(f"ValueError in Bayesian logistic prediction: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in Bayesian logistic prediction: {e}")
        raise


def predict_probability_naive_bayes(
    model: GaussianNB, X_new: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Predict probability using Naive Bayes classifier.

    Args:
        model (GaussianNB): Fitted Naive Bayes model
        X_new (np.ndarray): New feature values for prediction

    Returns:
        Dict[str, Union[float, np.ndarray]]: Prediction results

    Raises:
        ValueError: If input data is invalid
        Exception: For any unexpected errors during prediction

    Example:
        >>> pred = predict_probability_naive_bayes(nb_model, X_new)
        >>> print(f"Naive Bayes probability: {pred['mean_probability']:.3f}")
    """
    logger.debug("Making Naive Bayes prediction")

    try:
        if len(X_new) == 0:
            logger.error("Empty prediction input")
            raise ValueError("Prediction input cannot be empty")

        probabilities = model.predict_proba(X_new.reshape(1, -1))[:, 1]
        prob = probabilities[0]

        logger.debug(f"Naive Bayes prediction: {prob:.3f}")

        return {
            "mean_probability": prob,
            "std_probability": 0.1,  # Naive Bayes doesn't provide uncertainty estimates
            "credible_interval_95": [max(0, prob - 0.1), min(1, prob + 0.1)],
            "all_samples": np.array([prob]),
        }

    except ValueError as ve:
        logger.error(f"ValueError in Naive Bayes prediction: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in Naive Bayes prediction: {e}")
        raise


def predict_probability_bayesian_ridge(
    model: BayesianRidge, X_new: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Predict probability using Bayesian Ridge regression with sigmoid calibration.

    Args:
        model (BayesianRidge): Fitted Bayesian Ridge model
        X_new (np.ndarray): New feature values for prediction

    Returns:
        Dict[str, Union[float, np.ndarray]]: Prediction results

    Raises:
        ValueError: If input data is invalid
        Exception: For any unexpected errors during prediction

    Example:
        >>> pred = predict_probability_bayesian_ridge(br_model, X_new)
        >>> print(f"Bayesian Ridge probability: {pred['mean_probability']:.3f}")
    """
    logger.debug("Making Bayesian Ridge prediction")

    try:
        if len(X_new) == 0:
            logger.error("Empty prediction input")
            raise ValueError("Prediction input cannot be empty")

        # Bayesian Ridge gives continuous predictions, need to calibrate to probabilities
        prediction, std = model.predict(X_new.reshape(1, -1), return_std=True)

        # Simple calibration to probability (sigmoid function)
        probability = 1 / (1 + np.exp(-prediction[0]))
        uncertainty = std[0]

        logger.debug(
            f"Bayesian Ridge prediction: {probability:.3f} ± {uncertainty:.3f}"
        )

        return {
            "mean_probability": probability,
            "std_probability": uncertainty,
            "credible_interval_95": [
                max(0, probability - 2 * uncertainty),
                min(1, probability + 2 * uncertainty),
            ],
            "all_samples": np.array([probability]),
        }

    except ValueError as ve:
        logger.error(f"ValueError in Bayesian Ridge prediction: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in Bayesian Ridge prediction: {e}")
        raise


def create_prediction_scenarios(
    scaler: StandardScaler,
    county_encoder: LabelEncoder,
    owner_encoder: LabelEncoder,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Create example scenarios for model prediction testing.

    Args:
        scaler (StandardScaler): Fitted scaler for numerical features
        county_encoder (LabelEncoder): Fitted encoder for county categories
        owner_encoder (LabelEncoder): Fitted encoder for ownership categories
        feature_names (List[str]): Ordered list of feature names

    Returns:
        Dict[str, np.ndarray]: Dictionary of scenario arrays ready for prediction

    Raises:
        ValueError: If encoders don't contain required categories
        Exception: For any unexpected errors during scenario creation

    Example:
        >>> scenarios = create_prediction_scenarios(scaler, county_enc, owner_enc, features)
        >>> urban_prob = model.predict_proba([scenarios['urban']])
    """
    logger.debug("Creating prediction scenarios")

    try:
        scenarios = {}

        # Check if required categories exist in encoders
        counties = county_encoder.classes_
        owners = owner_encoder.classes_

        # Use first available categories if specific ones don't exist
        urban_county = "Nairobi City" if "Nairobi City" in counties else counties[0]
        rural_county = (
            "Kajiado"
            if "Kajiado" in counties
            else counties[-1]
            if len(counties) > 1
            else counties[0]
        )

        public_owner = next((o for o in owners if "Ministry" in o), owners[0])
        private_owner = next(
            (o for o in owners if "Private" in o),
            owners[-1] if len(owners) > 1 else owners[0],
        )

        logger.debug(f"Using counties: Urban={urban_county}, Rural={rural_county}")
        logger.debug(f"Using owners: Public={public_owner}, Private={private_owner}")

        # Scenario 1: High-density urban area
        urban_features = {
            "population_density_no_per_sq_km": 50000,
            "population_x": 300000,
            "land_area_sq_km": 5,
            "male": 150000,
            "female": 150000,
            "households": 100000,
            "average_household_size": 3.0,
            "urban_rural": 1,
            "county_encoded": county_encoder.transform([urban_county])[0],
            "owner_encoded": owner_encoder.transform([public_owner])[0],
        }

        # Scenario 2: Low-density rural area
        rural_features = {
            "population_density_no_per_sq_km": 100,
            "population_x": 50000,
            "land_area_sq_km": 500,
            "male": 25000,
            "female": 25000,
            "households": 15000,
            "average_household_size": 3.3,
            "urban_rural": 0,
            "county_encoded": county_encoder.transform([rural_county])[0],
            "owner_encoded": owner_encoder.transform([private_owner])[0],
        }

        # Convert to arrays in correct feature order
        for scenario_name, features_dict in [
            ("urban", urban_features),
            ("rural", rural_features),
        ]:
            # Handle missing features by using defaults
            scenario_array = []
            for feature in feature_names:
                if feature in features_dict:
                    scenario_array.append(features_dict[feature])
                else:
                    logger.warning(
                        f"Feature {feature} not found in scenario, using 0 as default"
                    )
                    scenario_array.append(0.0)

            scenario_array = np.array(scenario_array)

            # Scale numerical features (exclude encoded categorical features)
            numerical_mask = [
                feature not in ["county_encoded", "owner_encoded"]
                for feature in feature_names
            ]
            if any(numerical_mask):
                scenario_array[numerical_mask] = scaler.transform(
                    [scenario_array[numerical_mask]]
                )[0]

            scenarios[scenario_name] = scenario_array

        logger.debug(f"Created {len(scenarios)} prediction scenarios")
        return scenarios

    except ValueError as ve:
        logger.error(f"ValueError creating prediction scenarios: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error creating prediction scenarios: {e}")
        raise


def run_bayesian_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main function to run comprehensive Bayesian healthcare access analysis.

    This function orchestrates the entire analysis pipeline including feature creation,
    model training, prediction, and evaluation for multiple Bayesian approaches.

    Args:
        df (pd.DataFrame): Raw healthcare facilities DataFrame

    Returns:
        Dict[str, Any]: Comprehensive results including:
            - models: Dictionary of fitted models
            - predictions: Prediction results for test scenarios
            - X, y: Processed features and target
            - evaluation_metrics: Model performance metrics

    Raises:
        ValueError: If input DataFrame is invalid
        Exception: For any unexpected errors during analysis

    Example:
        >>> results = run_bayesian_analysis(facilities_df)
        >>> print(f"Analysis completed with {len(results['models'])} models")
        >>> for model_name, metrics in results['evaluation_metrics'].items():
        ...     print(f"{model_name} accuracy: {metrics['accuracy']:.3f}")
    """
    logger.info("=== Starting Bayesian Healthcare Access Analysis ===")

    try:
        if df.empty:
            logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame cannot be empty")

        logger.info(f"Analyzing {len(df)} healthcare facilities")

        # Step 1: Create features
        logger.info("Step 1: Creating features...")
        X, y, scaler, county_encoder, owner_encoder = create_features(df)

        # Step 2: Split data for evaluation
        logger.info("Step 2: Splitting data for training and testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(
            f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples"
        )

        # Step 3: Try different Bayesian models
        models = {}
        predictions = {}
        evaluation_metrics = {}

        # Method 1: Naive Bayes (always available)
        logger.info("Step 3a: Building Naive Bayes model...")
        try:
            nb_model = build_naive_bayes_model(X_train.values, y_train.values)
            models["naive_bayes"] = nb_model

            # Evaluate on test set
            y_pred_nb = nb_model.predict(X_test.values)
            y_pred_proba_nb = nb_model.predict_proba(X_test.values)[:, 1]

            evaluation_metrics["naive_bayes"] = {
                "accuracy": accuracy_score(y_test, y_pred_nb),
                "auc_roc": roc_auc_score(y_test, y_pred_proba_nb)
                if len(np.unique(y_test)) > 1
                else 0.5,
            }
            logger.info(
                f"✓ Naive Bayes - Accuracy: {evaluation_metrics['naive_bayes']['accuracy']:.3f}"
            )

        except Exception as e:
            logger.error(f"✗ Naive Bayes failed: {e}")

        # Method 2: Bayesian Ridge
        logger.info("Step 3b: Building Bayesian Ridge model...")
        try:
            br_model = build_bayesian_ridge_model(X_train.values, y_train.values)
            models["bayesian_ridge"] = br_model

            # Evaluate on test set (convert continuous predictions to binary)
            y_pred_br_cont = br_model.predict(X_test.values)
            y_pred_br = (y_pred_br_cont > 0.5).astype(int)
            y_pred_proba_br = 1 / (
                1 + np.exp(-y_pred_br_cont)
            )  # Sigmoid transformation

            evaluation_metrics["bayesian_ridge"] = {
                "accuracy": accuracy_score(y_test, y_pred_br),
                "auc_roc": roc_auc_score(y_test, y_pred_proba_br)
                if len(np.unique(y_test)) > 1
                else 0.5,
            }
            logger.info(
                f"✓ Bayesian Ridge - Accuracy: {evaluation_metrics['bayesian_ridge']['accuracy']:.3f}"
            )

        except Exception as e:
            logger.error(f"✗ Bayesian Ridge failed: {e}")

        # Method 3: Bayesian Logistic (statsmodels) - most computationally intensive
        logger.info("Step 3c: Building Bayesian Logistic model...")
        try:
            bl_model, bl_result = build_bayesian_logistic_model(X_train, y_train)
            models["bayesian_logistic"] = (bl_model, bl_result)

            # Evaluate on test set
            X_test_const = sm.add_constant(X_test)
            y_pred_proba_bl = bl_result.predict(X_test_const).mean(axis=0)
            y_pred_bl = (y_pred_proba_bl > 0.5).astype(int)

            evaluation_metrics["bayesian_logistic"] = {
                "accuracy": accuracy_score(y_test, y_pred_bl),
                "auc_roc": roc_auc_score(y_test, y_pred_proba_bl)
                if len(np.unique(y_test)) > 1
                else 0.5,
            }
            logger.info(
                f"✓ Bayesian Logistic - Accuracy: {evaluation_metrics['bayesian_logistic']['accuracy']:.3f}"
            )

        except Exception as e:
            logger.error(f"✗ Bayesian Logistic failed: {e}")

        # Step 4: Make predictions on scenarios
        logger.info("Step 4: Making predictions on test scenarios...")
        feature_names = X.columns.tolist()
        scenarios = create_prediction_scenarios(
            scaler, county_encoder, owner_encoder, feature_names
        )

        # Predict with each successful model
        for model_name, model in models.items():
            logger.info(f"--- Predictions using {model_name} ---")

            try:
                if model_name == "naive_bayes":
                    urban_pred = predict_probability_naive_bayes(
                        model, scenarios["urban"]
                    )
                    rural_pred = predict_probability_naive_bayes(
                        model, scenarios["rural"]
                    )
                elif model_name == "bayesian_ridge":
                    urban_pred = predict_probability_bayesian_ridge(
                        model, scenarios["urban"]
                    )
                    rural_pred = predict_probability_bayesian_ridge(
                        model, scenarios["rural"]
                    )
                elif model_name == "bayesian_logistic":
                    urban_pred = predict_probability_bayesian_logistic(
                        model[1], scenarios["urban"], feature_names
                    )
                    rural_pred = predict_probability_bayesian_logistic(
                        model[1], scenarios["rural"], feature_names
                    )

                logger.info(
                    f"Urban area probability: {urban_pred['mean_probability']:.3f} ± {urban_pred['std_probability']:.3f}"
                )
                logger.info(
                    f"Rural area probability: {rural_pred['mean_probability']:.3f} ± {rural_pred['std_probability']:.3f}"
                )

                predictions[model_name] = {"urban": urban_pred, "rural": rural_pred}

            except Exception as e:
                logger.error(f"Error making predictions with {model_name}: {e}")

        # Step 5: Summary and insights
        logger.info("Step 5: Generating analysis summary...")

        if evaluation_metrics:
            logger.info("=== Model Performance Summary ===")
            for model_name, metrics in evaluation_metrics.items():
                logger.info(
                    f"{model_name:15} - Accuracy: {metrics['accuracy']:.3f}, AUC-ROC: {metrics['auc_roc']:.3f}"
                )

        if predictions:
            logger.info("=== Scenario Predictions Summary ===")
            for scenario in ["urban", "rural"]:
                logger.info(f"{scenario.capitalize()} scenario predictions:")
                for model_name, pred_dict in predictions.items():
                    if scenario in pred_dict:
                        prob = pred_dict[scenario]["mean_probability"]
                        logger.info(f"  {model_name:15}: {prob:.3f}")

        # Compile results
        results = {
            "models": models,
            "predictions": predictions,
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "evaluation_metrics": evaluation_metrics,
            "scaler": scaler,
            "county_encoder": county_encoder,
            "owner_encoder": owner_encoder,
            "feature_names": feature_names,
        }

        logger.info("=== Analysis Complete ===")
        logger.info(f"Successfully trained {len(models)} models")
        logger.info(
            "Bayesian models provide probabilistic estimates of specialized care access"
        )

        return results

    except ValueError as ve:
        logger.error(f"ValueError in Bayesian analysis: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Bayesian analysis: {e}")
        raise


def simple_bayesian_simulation(
    X: pd.DataFrame, y: pd.Series, n_simulations: int = 1000
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform simple Bayesian simulation using bootstrap sampling.

    This function provides an alternative Bayesian approach using bootstrap resampling
    to estimate parameter uncertainty when MCMC methods are not feasible.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        n_simulations (int, optional): Number of bootstrap simulations. Defaults to 1000.

    Returns:
        Dict[str, Union[float, np.ndarray]]: Bootstrap simulation results including:
            - mean_probability: Average predicted probability across simulations
            - std_probability: Standard deviation of probability estimates
            - credible_interval_95: 95% credible interval from bootstrap distribution

    Raises:
        ValueError: If input data is invalid or n_simulations <= 0
        Exception: For any unexpected errors during simulation

    Example:
        >>> bootstrap_results = simple_bayesian_simulation(X, y, n_simulations=500)
        >>> print(f"Bootstrap probability: {bootstrap_results['mean_probability']:.3f}")
    """
    logger.info("Running simple Bayesian simulation with bootstrap sampling...")

    try:
        if X.empty or y.empty:
            logger.error("Input data is empty")
            raise ValueError("Input data cannot be empty")

        if n_simulations <= 0:
            logger.error(f"Invalid number of simulations: {n_simulations}")
            raise ValueError("Number of simulations must be positive")

        logger.debug(f"Running {n_simulations} bootstrap simulations")

        n_samples = len(y)
        probabilities = []
        successful_sims = 0

        for i in range(n_simulations):
            try:
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]

                # Simple logistic regression on bootstrap sample
                X_with_const = sm.add_constant(X_boot)
                model = sm.Logit(y_boot, X_with_const)
                result = model.fit(disp=0, maxiter=100)

                # Predict on mean feature values
                mean_features = sm.add_constant(X.mean().values.reshape(1, -1))
                prob = result.predict(mean_features)[0]
                probabilities.append(prob)
                successful_sims += 1

            except Exception as e:
                if i % 100 == 0:  # Log occasional failures
                    logger.debug(f"Bootstrap iteration {i} failed: {e}")
                continue

        if successful_sims == 0:
            logger.error("All bootstrap simulations failed")
            raise Exception("All bootstrap simulations failed")

        probabilities = np.array(probabilities)

        logger.info(
            f"Bootstrap simulation completed: {successful_sims}/{n_simulations} successful"
        )

        result = {
            "mean_probability": np.mean(probabilities),
            "std_probability": np.std(probabilities),
            "credible_interval_95": np.percentile(probabilities, [2.5, 97.5]),
            "successful_simulations": successful_sims,
            "all_probabilities": probabilities,
        }

        logger.debug(
            f"Bootstrap results - Mean: {result['mean_probability']:.3f}, "
            f"Std: {result['std_probability']:.3f}"
        )

        return result

    except ValueError as ve:
        logger.error(f"ValueError in bootstrap simulation: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in bootstrap simulation: {e}")
        raise


def evaluate_model_performance(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    detailed: bool = True,
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    Evaluate performance of trained Bayesian models on test data.

    Args:
        models (Dict[str, Any]): Dictionary of trained models
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target values
        detailed (bool, optional): Whether to include detailed metrics. Defaults to True.

    Returns:
        Dict[str, Dict[str, Union[float, np.ndarray]]]: Performance metrics for each model

    Raises:
        ValueError: If input data is invalid
        Exception: For any unexpected errors during evaluation

    Example:
        >>> metrics = evaluate_model_performance(models, X_test, y_test)
        >>> for model_name, model_metrics in metrics.items():
        ...     print(f"{model_name} accuracy: {model_metrics['accuracy']:.3f}")
    """
    logger.info("Evaluating model performance on test data...")

    try:
        if X_test.empty or y_test.empty:
            logger.error("Test data is empty")
            raise ValueError("Test data cannot be empty")

        performance_metrics = {}

        for model_name, model in models.items():
            logger.debug(f"Evaluating {model_name}...")

            try:
                if model_name == "naive_bayes":
                    y_pred = model.predict(X_test.values)
                    y_pred_proba = model.predict_proba(X_test.values)[:, 1]

                elif model_name == "bayesian_ridge":
                    y_pred_cont = model.predict(X_test.values)
                    y_pred = (y_pred_cont > 0.5).astype(int)
                    y_pred_proba = 1 / (1 + np.exp(-y_pred_cont))

                elif model_name == "bayesian_logistic":
                    X_test_const = sm.add_constant(X_test)
                    y_pred_proba = model[1].predict(X_test_const).mean(axis=0)
                    y_pred = (y_pred_proba > 0.5).astype(int)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc_roc = (
                    roc_auc_score(y_test, y_pred_proba)
                    if len(np.unique(y_test)) > 1
                    else 0.5
                )

                metrics = {"accuracy": accuracy, "auc_roc": auc_roc}

                if detailed:
                    # Additional detailed metrics
                    from sklearn.metrics import precision_score, recall_score, f1_score

                    metrics.update(
                        {
                            "precision": precision_score(
                                y_test, y_pred, zero_division=0
                            ),
                            "recall": recall_score(y_test, y_pred, zero_division=0),
                            "f1_score": f1_score(y_test, y_pred, zero_division=0),
                            "confusion_matrix": confusion_matrix(y_test, y_pred),
                            "classification_report": classification_report(
                                y_test, y_pred, output_dict=True
                            ),
                        }
                    )

                performance_metrics[model_name] = metrics
                logger.info(
                    f"✓ {model_name} - Accuracy: {accuracy:.3f}, AUC-ROC: {auc_roc:.3f}"
                )

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                performance_metrics[model_name] = {"error": str(e)}

        logger.info(
            f"Performance evaluation completed for {len(performance_metrics)} models"
        )
        return performance_metrics

    except ValueError as ve:
        logger.error(f"ValueError in model evaluation: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise


# Example usage function
def main_analysis_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute the complete Bayesian healthcare analysis pipeline.

    Args:
        df (pd.DataFrame): Healthcare facilities dataset

    Returns:
        Dict[str, Any]: Complete analysis results

    Example:
        >>> results = main_analysis_pipeline(merged_df)
        >>> print("Analysis complete!")
    """
    logger.info("Starting main analysis pipeline...")

    try:
        # Run the main Bayesian analysis
        results = run_bayesian_analysis(df)

        # Optional: Run bootstrap simulation for additional validation
        if "X" in results and "y" in results:
            logger.info("Running additional bootstrap validation...")
            bootstrap_results = simple_bayesian_simulation(results["X"], results["y"])
            results["bootstrap_validation"] = bootstrap_results

        # Enhanced evaluation if test data is available
        if all(key in results for key in ["models", "X_test", "y_test"]):
            logger.info("Running detailed model evaluation...")
            detailed_metrics = evaluate_model_performance(
                results["models"], results["X_test"], results["y_test"], detailed=True
            )
            results["detailed_evaluation"] = detailed_metrics

        logger.info("=== Analysis Pipeline Complete ===")
        return results

    except Exception as e:
        logger.error(f"Error in main analysis pipeline: {e}")
        raise
