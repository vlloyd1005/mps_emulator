# # Author: Victoria Lloyd (2025) & V. Miranda
# import os
# import numpy as np
# import joblib
# from typing import Dict, Any, List, Tuple, Optional
# import logging
# from pathlib import Path
# import sys
# import train_utils_pk_emulator as utils
# sys.modules['train_utils_pk_emulator'] = utils
# from train_utils_pk_emulator import CustomActivationLayer
# from keras.losses import MeanSquaredError
# import time
# import tensorflow as tf


# # Set up logging for the module
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# # --- Path Management ---
# def _get_project_root() -> Path:
#     """Returns the root directory Path object, relative to this file."""
#     return Path(__file__).resolve().parent

# # --- Dependency Guards ---
# ROOT = _get_project_root()
# try:

#     from tensorflow import keras
#     from colossus.cosmology import cosmology as Cosmo
#     # Issues installing symbolic_pofk as a package, including a local copy in the emulator
#     import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk"); from symbolic_pofk.linear_VM import plin_emulated, get_approximate_D, growth_correction_R
    
#     _DEPENDENCIES_LOADED = True
# except ImportError as e:
#     logging.error(f"FATAL ERROR: A required dependency could not be imported. Please ensure all dependencies are installed.")
#     logging.error(f"Missing component: {e.name}")
#     logging.error(f"If running this package locally, ensure the symbolic_pofk library is accessible.")
#     _DEPENDENCIES_LOADED = False

# try:
#     from sklearn.decomposition import PCA 
#     from sklearn.preprocessing import StandardScaler
# except ImportError:
#     pass

# # --- CORE EMULATOR CLASS ---
# class PkEmulator:
#     """
#     Cosmology Emulator for the Matter Power Spectrum P(k, z).
#     Encapsulates models and handles I/O robustly.
    
#     Supports multiple cosmological models (lcdm, w0wacdm) and 
#     nonlinear prescriptions (linear, halofit, mead2020, mead2020_feedback).
#     """

#     # --- Configuration Constants (Class Attributes) ---
#     N_PCS = 25          # Number of k-space PCs per redshift
#     N_K_MODES = 500     # Number of k modes
    
#     # Fixed k and z grids
#     K_MODES = np.logspace(-5.1, 2, N_K_MODES)
#     Z_MODES = np.concatenate((
#         np.linspace(0, 3, 33, endpoint=False),
#         np.linspace(3, 10, 7, endpoint=False),
#         np.linspace(10, 50, 12)
#     ))
#     N_ZS = len(Z_MODES)  # Number of redshift bins
    
#     # Valid model types and nonlinear prescriptions
#     VALID_MODELS = ['lcdm', 'w0wacdm']
#     VALID_NONLINEAR = ['lin', 'halofit', 'mead2020', 'mead2020_feedback']
    
#     def __init__(
#         self, 
#         base_model_path: str = "models",
#         base_metadata_path: str = "metadata", 
#         model_type: str = "w0wacdm",
#         nonlinear_prescription: str = "lin",
#         num_batches: int = 10
#     ):
#         """
#         Initializes the emulator by loading all necessary models and metadata.
        
#         Args:
#             base_model_path: Base directory containing model files
#             base_metadata_path: Base directory containing metadata subdirectories
#             model_type: Cosmological model type ('lcdm' or 'w0wacdm')
#             nonlinear_prescription: Nonlinear prescription ('linear', 'halofit', 
#                                    'mead2020', or 'mead2020_feedback')
#             num_batches: Number of batches used in training
        
#         Note:
#             All models expect 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
#             For LCDM, the user should pass w0=-1.0 and wa=0.0 (standard LCDM values).
#         """
        
#         if not _DEPENDENCIES_LOADED:
#             raise RuntimeError("Cannot initialize PkEmulator due to missing dependencies.")

#         # Validate inputs
#         if model_type not in self.VALID_MODELS:
#             raise ValueError(f"model_type must be one of {self.VALID_MODELS}, got '{model_type}'")
#         if nonlinear_prescription not in self.VALID_NONLINEAR:
#             raise ValueError(f"nonlinear_prescription must be one of {self.VALID_NONLINEAR}, got '{nonlinear_prescription}'")

#         self.model_type = model_type
#         self.nonlinear_prescription = nonlinear_prescription
        
#         logging.info(f"[PkEmulator] Initializing emulator with model_type='{model_type}', "
#                     f"nonlinear_prescription='{nonlinear_prescription}'")
        
#         # Paths are relative to the package root
#         self.MODEL_DIR = ROOT / base_model_path
#         self.BASE_METADATA_DIR = ROOT / base_metadata_path
#         self.NUM_BATCHES = num_batches
        
#         # Determine metadata directory based on model type and nonlinear prescription
#         self.METADATA_DIR = self._get_metadata_directory()
        
#         # Load the core components using Path objects
#         try:
#             self.param_scaler = joblib.load(self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches")
#             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")
            
#             # Construct model filename based on model_type and nonlinear_prescription
#             model_filename = self._get_model_filename()
            
#             self.model = keras.models.load_model(
#                 self.MODEL_DIR / model_filename,
#                 custom_objects={
#                     "CustomActivationLayer": CustomActivationLayer,
#                     "mse": MeanSquaredError()
#                 },
#             )

#             # Create compiled inference function with XLA
#             @tf.function(jit_compile=False)
#             def compiled_inference(x):
#                 return self.model(x, training=False)
            
#             self._compiled_inference = compiled_inference

#             # Initialize empty dictionaries for PCA and Scaler objects
#             self.PCAS: Dict[float, PCA] = {}
#             self.SCALERS: Dict[float, StandardScaler] = {}
#             self._pcas_loaded = False

#             # Pre-computed inverse transform matrices
#             self.INVERSE_TRANSFORM_MATRICES = None
#             self.INVERSE_TRANSFORM_OFFSETS = None

#             # Load PCA/Scalers
#             self._load_pcas_and_scalers()
            
#             # Warm up both the neural network AND the compiled function
#             logging.info("[PkEmulator] Warming up neural network...")
            
#             # All models expect 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa]
#             dummy_params = self.param_scaler.transform(np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, 0.0]], dtype=np.float32))
#             dummy_tf = tf.constant(dummy_params, dtype=tf.float32)
            
#             # Warm up compiled function (first call triggers XLA compilation)
#             _ = self._compiled_inference(dummy_tf)
#             # Second call to ensure compilation is complete
#             _ = self._compiled_inference(dummy_tf)

#             logging.info("[PkEmulator] Core models loaded successfully.")

#         except FileNotFoundError as e:
#             logging.error(f"CRITICAL: Required model or metadata file not found: {e.filename}")
#             logging.warning(f"Please ensure the '{self.METADATA_DIR}' directory and '{self.MODEL_DIR}' "
#                           "contain the required files.")
#             raise
    
#     def _get_metadata_directory(self) -> Path:
#         """
#         Determines the metadata directory based on model_type and nonlinear_prescription.
        
#         Directory naming convention:
#         - LCDM + linear: metadata_lcdm
#         - w0waCDM + linear: metadata_w0wacdm
#         - w0waCDM + halofit: metadata_halofit_w0wa
#         - w0waCDM + mead2020: metadata_mead2020
#         - w0waCDM + mead2020_feedback: metadata_mead2020_feedback
        
#         Returns:
#             Path: Path to the appropriate metadata directory
#         """
#         dir_name = f"metadata_{self.model_type}_{self.nonlinear_prescription}"
#         # if self.model_type == 'lcdm':
#         #     if self.nonlinear_prescription == 'lin':
#         #         dir_name = "metadata_lcdm"
#         #     else:
#         #         # LCDM with nonlinear prescriptions
#         #         dir_name = f"metadata_{self.nonlinear_prescription}_lcdm"
#         # else:  # w0wacdm
#         #     if self.nonlinear_prescription == 'lin':
#         #         dir_name = f"metadata_{self.nonlinear_prescription}"
        
#         metadata_dir = self.BASE_METADATA_DIR / dir_name
#         logging.info(f"[PkEmulator] Using metadata directory: {metadata_dir}")
#         return metadata_dir
    
#     def _get_model_filename(self) -> str:
#         """
#         Constructs the model filename based on model_type and nonlinear_prescription.
        
#         Filename convention: emulator_hmpc_{model_type}_{nonlinear_prescription}.h5
#         For linear prescription, no suffix is added.
        
#         Returns:
#             str: Model filename
#         """
#         # if self.nonlinear_prescription == 'lin':
#         #     filename = f"emulator_{self.model_type}.h5"
#         # else:
#         filename = f"emulator_{self.model_type}_{self.nonlinear_prescription}.h5"
        
#         logging.info(f"[PkEmulator] Using model file: {filename}")
#         return filename
    
#     def _load_pcas_and_scalers(self):
#         """
#         Eager loading of PCA and Scaler objects with pre-computation of inverse transforms.
        
#         Only loads these if they haven't been loaded yet. This is called
#         automatically during __init__ to pre-compute inverse transformation matrices.
        
#         Pre-computes the combined inverse PCA + inverse scaling
#         transformation matrices for maximum performance.
#         """
#         if self._pcas_loaded:
#             return  # Already loaded
        
#         logging.info("[PkEmulator] Loading PCA and Scaler objects...")
        
#         try:
#             # Load individual PCA and Scaler objects
#             for z in self.Z_MODES:
#                 z_key = float(f"{z:.3f}")
#                 self.PCAS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
#                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")
            
#             logging.info("[PkEmulator] PCA and Scaler objects loaded successfully.")
#             logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")
            
#             # Pre-compute combined inverse transformation matrices
#             # This combines: (PCs @ PCA.components_ + PCA.mean_) * std + mean
#             # Into a single matrix multiplication + offset addition
            
#             inverse_matrices = []
#             inverse_offsets = []
            
#             for z in self.Z_MODES:
#                 z_key = float(f"{z:.3f}")
#                 pca = self.PCAS[z_key]
#                 scaler = self.SCALERS[z_key]
                
#                 # Get the scaling factor (handle both sklearn and custom Scaler)
#                 if hasattr(scaler, 'scale_'):
#                     scale = scaler.scale_  # sklearn StandardScaler
#                     mean = scaler.mean_
#                 elif hasattr(scaler, 'std'):
#                     scale = scaler.std     # Custom Scaler
#                     mean = scaler.mean
#                 else:
#                     raise AttributeError(f"Scaler object has neither 'scale_' nor 'std' attribute")
                
#                 # Combined transformation matrix
#                 # Shape: (N_PCS, N_K_MODES)
#                 # Each row of PCA.components_ is scaled element-wise by scaler.std
#                 combined_matrix = pca.components_ * scale[None, :]
                
#                 # Combined offset vector
#                 # Shape: (N_K_MODES,)
#                 # This is: PCA.mean_ * scaler.std + scaler.mean
#                 combined_offset = pca.mean_ * scale + mean
                
#                 inverse_matrices.append(combined_matrix)
#                 inverse_offsets.append(combined_offset)
            
#             # Stack into arrays for vectorized operations
#             # Shape: (N_ZS, N_PCS, N_K_MODES)
#             self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
            
#             # Shape: (N_ZS, N_K_MODES)
#             self.INVERSE_TRANSFORM_OFFSETS = np.stack(inverse_offsets, axis=0).astype(np.float32)
            
#             self._pcas_loaded = True
#             logging.info("[PkEmulator] Inverse transformation matrices pre-computed successfully.")
            
#         except FileNotFoundError as e:
#             logging.error(f"CRITICAL: Required PCA/Scaler file not found: {e.filename}")
#             logging.warning(f"Please ensure the '{self.METADATA_DIR}' directory contains all PCA and scaler files.")
#             raise
    
#     def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
#         """
#         Computes the analytical P_lin(k, z) approximation.
        
#         This function uses Colossus to calculate the growth factors.
        
#         Args:
#             params: 1D array of cosmological parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
#                    For LCDM models, w0=-1.0 and wa=0.0 should be passed.
            
#         Returns:
#             np.ndarray: P_lin(k, z) array of shape (N_ZS, N_K_MODES).
#         """
#         # Destructure parameters (always 7 parameters)
#         As, ns, H0_in, Ob, Om, w0, wa = params
        
#         h = H0_in / 100.0 # Convert H0 to h
        
#         # Compute fiducial P(k) at z=0
#         # Convert k for plin_emulated: 1/Mpc -> h/Mpc
#         k_for_plin = self.K_MODES / h
#         pk_fid = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)
        
#         a_array = 1.0/(self.Z_MODES + 1)
#         D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, 
#                                ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
#         Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, 
#                                ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
#         R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, 
#                                  ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
#         Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, 
#                                  ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
#         growth_factors = (Dz/D0)*(Dz/D0)*(Rz/R0)
        
#         # pk_fid[None, :] broadcasting works cleanly: (1, K) * (Z, 1) = (Z, K)
#         result = pk_fid[None, :] * growth_factors[:, None]
#         result = result / h**3
#         return result.astype(np.float32)

#     def _predict_fracs_all_z(self, params: np.ndarray) -> np.ndarray:
#         """Optimized NN inference using compiled TensorFlow function.
#         Performs the full NN prediction and PCA reconstructions,
#         optimized using compiled TensorFlow function.
        
#         This method requires PCA and Scaler objects, which are loaded
#         at initialization to avoid unnecessary loading during evaluation.
        
#         Args:
#             params: Normalized 2D array of parameters (1, N_params).
            
#         Returns:
#             np.ndarray: Final predicted log fractional differences, shape (N_ZS, N_K_MODES).
#         """
#         # Convert to TF tensor and use compiled inference
#         params_tf = tf.constant(params, dtype=tf.float32)

#         # Predict T-components
#         t_comps_pred = self._compiled_inference(params_tf).numpy()
        
#         # Inverse T-components to flat k-PCs (shape 1, N_ZS * N_PCS)
#         pcs_flat = self.t_comp_pca.inverse_transform(t_comps_pred).astype(np.float32)

#         # Reshape to individual redshifts (shape N_ZS, N_PCS)
#         # We drop the single batch dimension (axis 0) since we only have one cosmology
#         pcs_pred_z_stack = pcs_flat.reshape(self.N_ZS, self.N_PCS)
        
#         reconstructed_fracs = (
#             np.einsum('zp,zpk->zk', pcs_pred_z_stack, self.INVERSE_TRANSFORM_MATRICES) +
#             self.INVERSE_TRANSFORM_OFFSETS
#         )
        
#         return reconstructed_fracs.astype(np.float32)

#     def get_pks(self, params: List[float], use_syren: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Returns P(k, z) for all emulator redshifts for a given cosmology.
        
#         Args:
#             params: List or 1D array of 7 cosmological parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
#                    For LCDM models, pass w0=-1.0 and wa=0.0.
#             use_syren: Optional flag to bypass emulator corrections.
#                       - None (default): Apply emulator corrections (full emulator)
#                       - True: Bypass emulator, return only symbolic approximation
            
#         Returns:
#             Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
#         """
#         # Validate parameter count (always expect 7 parameters)
#         params_array = np.array(params, dtype=np.float32)
#         if len(params_array) != 7:
#             raise ValueError(f"Expected 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa], "
#                            f"got {len(params_array)}. For LCDM, pass w0=-1.0, wa=0.0")
        
#         # Normalize cosmological parameters
#         # Ensure input is a 2D array (1, N_params) for the scaler/model
#         x_norm = self.param_scaler.transform(params_array.reshape(1, -1))
        
#         # Compute MPS approximation (symbolic P(k) - always needed)
#         pk_mps = self._compute_mps_approximation(params_array)

#         # Decide whether to apply emulator corrections
#         if use_syren is True:
#             # Bypass emulator - return only symbolic approximation
#             pks = pk_mps
#         else:
#             # Default behavior: apply emulator corrections
#             # Generate predicted fractional differences (shape N_ZS, N_K_MODES)
#             frac = np.exp(self._predict_fracs_all_z(x_norm), dtype=np.float32)
            
#             # Full emulated P(k, z)
#             pks = (frac * pk_mps).astype(np.float32)

#         # Return the k-modes, z-modes, and the P(k,z) array
#         return self.K_MODES, self.Z_MODES, pks



# # --- Public Module-Level Interface ---

# # Global emulator instances (lazy initialization)
# _pk_emulator_instances: Dict[Tuple[str, str], PkEmulator] = {}

# def get_emulator(
#     model_type: str = "w0wacdm",
#     nonlinear_prescription: str = "lin",
#     base_model_path: str = "models",
#     base_metadata_path: str = "metadata",
#     num_batches: int = 10
# ) -> PkEmulator:
#     """
#     Get or create an emulator instance with specified configuration.
    
#     Instances are cached to avoid reloading the same model multiple times.
    
#     Args:
#         model_type: Cosmological model type ('lcdm' or 'w0wacdm')
#         nonlinear_prescription: Nonlinear prescription ('linear', 'halofit', 
#                                'mead2020', or 'mead2020_feedback')
#         base_model_path: Base directory containing model files
#         base_metadata_path: Base directory containing metadata subdirectories
#         num_batches: Number of batches used in training
    
#     Returns:
#         PkEmulator: Configured emulator instance
#     """
#     if not _DEPENDENCIES_LOADED:
#         raise RuntimeError("Cannot create PkEmulator due to missing dependencies.")
    
#     # Create cache key
#     cache_key = (model_type, nonlinear_prescription)
    
#     # Return cached instance if available
#     if cache_key in _pk_emulator_instances:
#         logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
#         return _pk_emulator_instances[cache_key]
    
#     # Create new instance
#     try:
#         logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
#         emulator = PkEmulator(
#             base_model_path=base_model_path,
#             base_metadata_path=base_metadata_path,
#             model_type=model_type,
#             nonlinear_prescription=nonlinear_prescription,
#             num_batches=num_batches
#         )
#         _pk_emulator_instances[cache_key] = emulator
#         return emulator
#     except Exception as e:
#         logging.error(f"Failed to create emulator for {cache_key}: {e}")
#         raise

# def get_pks(
#     params: List[float], 
#     use_syren: bool = None,
#     model_type: str = "w0wacdm",
#     nonlinear_prescription: str = "lin"
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Module-level function to get P(k, z). This is the streamlined public interface.
    
#     Args:
#         params: List or 1D array of 7 cosmological parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa].
#                For LCDM models, pass w0=-1.0 and wa=0.0 (standard LCDM values).
#         use_syren: Optional flag to bypass emulator corrections.
#                   - None (default): Apply emulator corrections (full emulator)
#                   - True: Bypass emulator, return only symbolic approximation
#         model_type: Cosmological model type ('lcdm' or 'w0wacdm')
#         nonlinear_prescription: Nonlinear prescription ('linear', 'halofit', 
#                                'mead2020', or 'mead2020_feedback')
            
#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: (k_modes, z_modes, P(k,z)).
#     """
#     emulator = get_emulator(model_type=model_type, nonlinear_prescription=nonlinear_prescription)
#     return emulator.get_pks(params, use_syren=use_syren)

# Author: Victoria Lloyd (2025) & V. Miranda
import os
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys
import train_utils_pk_emulator as utils
sys.modules['train_utils_pk_emulator'] = utils
from train_utils_pk_emulator import CustomActivationLayer, TComponentScaler
from keras.losses import MeanSquaredError, Huber
import tensorflow as tf


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ----------------------------------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------------------------------

def _get_project_root() -> Path:
    """Returns the directory containing this file."""
    return Path(__file__).resolve().parent

ROOT = _get_project_root()

# ----------------------------------------------------------------------------------------------------
# Dependency guard
# ----------------------------------------------------------------------------------------------------

try:
    from tensorflow import keras
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
    from symbolic_pofk.linear_VM import plin_emulated, get_approximate_D, growth_correction_R
    _DEPENDENCIES_LOADED = True
except ImportError as e:
    logging.error("FATAL ERROR: A required dependency could not be imported.")
    logging.error(f"Missing component: {e.name}")
    logging.error("If running locally, ensure the symbolic_pofk library is accessible.")
    _DEPENDENCIES_LOADED = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass


# ----------------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------------

NL_TYPE_REGISTRY = {
    "lin":                      ("pklin",                           "Linear P(k)"),
    "halofit":                  ("pknonlin",                        "Non-linear P(k) via HaloFit"),
    "mead2020":                 ("mead2020_pknonlin",               "Non-linear P(k) via HMcode Mead2020"),
    "mead2020_feedback":        ("mead2020_feedback_pknonlin",      "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
    "mead2020_feedback_Tfree":  ("mead2020_feedback_Tfree_pknonlin","Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
}

VALID_NL_TYPES_BY_PRIOR = {
    "expanded":    {"lin", "halofit"},
    "constrained": set(NL_TYPE_REGISTRY.keys()),
}

VALID_COSMO_TYPES = {"lcdm", "w0wacdm"}


def _validate_config(cosmo_type: str, prior_type: str, nl_type: str) -> None:
    if cosmo_type not in VALID_COSMO_TYPES:
        raise ValueError(
            f"Unknown cosmo_type '{cosmo_type}'. Must be one of: {sorted(VALID_COSMO_TYPES)}"
        )
    if prior_type not in VALID_NL_TYPES_BY_PRIOR:
        raise ValueError(
            f"Unknown prior_type '{prior_type}'. Must be one of: {sorted(VALID_NL_TYPES_BY_PRIOR)}"
        )
    if nl_type not in NL_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown nl_type '{nl_type}'. Must be one of: {sorted(NL_TYPE_REGISTRY)}"
        )
    if nl_type not in VALID_NL_TYPES_BY_PRIOR[prior_type]:
        raise ValueError(
            f"nl_type '{nl_type}' is not available for prior_type='{prior_type}'. "
            f"Valid choices: {sorted(VALID_NL_TYPES_BY_PRIOR[prior_type])}"
        )
    if cosmo_type == "lcdm" and prior_type == "expanded":
        raise ValueError("prior_type='expanded' is only supported for cosmo_type='w0wacdm'.")


def _metadata_tag(cosmo_type: str, prior_type: str, nl_type: str) -> str:
    return f"{cosmo_type}_{prior_type}_{nl_type}"


def _model_filename(cosmo_type: str, prior_type: str, nl_type: str) -> str:
    return f"emulator_{_metadata_tag(cosmo_type, prior_type, nl_type)}.h5"


# ----------------------------------------------------------------------------------------------------
# Core emulator class
# ----------------------------------------------------------------------------------------------------

class PkEmulator:
    """
    Cosmology emulator for the matter power spectrum P(k, z).

    Supports multiple cosmological models, prior widths, and nonlinear prescriptions.
    All models expect 7 parameters: [10^9 A_s, ns, H0, Ob, Om, w0, wa].
    For LCDM, pass w0=-1.0 and wa=0.0.

    Parameters
    ----------
    cosmo_type : str
        Cosmological model: ``'w0wacdm'`` or ``'lcdm'``.
    prior_type : str
        Prior width used during training: ``'expanded'`` or ``'constrained'``.
    nl_type : str
        Nonlinear prescription. See NL_TYPE_REGISTRY for valid options.
    base_model_path : str
        Directory containing model .h5 files, relative to this file.
    base_metadata_path : str
        Parent directory of all metadata subdirectories, relative to this file.
    num_batches : int
        Number of training batches — used to locate the param scaler file.
    """

    N_PCS     = 25
    N_K_MODES = 500

    K_MODES = np.logspace(-5.1, 2, N_K_MODES)
    Z_MODES = np.concatenate((
        np.linspace(0,  3,  33, endpoint=False),
        np.linspace(3,  10,  7, endpoint=False),
        np.linspace(10, 50, 12),
    ))
    N_ZS = len(Z_MODES)

    def __init__(
        self,
        cosmo_type: str = "w0wacdm",
        prior_type: str = "constrained",
        nl_type: str = "lin",
        base_model_path: str = "models",
        base_metadata_path: str = "metadata",
        num_batches: int = 15,
    ):
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialise PkEmulator: missing dependencies.")

        _validate_config(cosmo_type, prior_type, nl_type)

        self.cosmo_type = cosmo_type
        self.prior_type = prior_type
        self.nl_type    = nl_type

        logging.info(
            f"[PkEmulator] Initialising: cosmo_type='{cosmo_type}', "
            f"prior_type='{prior_type}', nl_type='{nl_type}'"
        )

        self.MODEL_DIR    = ROOT / base_model_path
        self.METADATA_DIR = (
            ROOT / base_metadata_path
            / f"metadata_{_metadata_tag(cosmo_type, prior_type, nl_type)}"
        )
        self.NUM_BATCHES  = num_batches

        logging.info(f"[PkEmulator] Metadata directory: {self.METADATA_DIR}")

        try:
            # --- Cosmological parameter scaler ---
            self.param_scaler = joblib.load(
                self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches"
            )

            # --- tPCA (maps flat PCA coefficients <-> t-components) ---
            self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")

            # --- [NEW] t-component scaler (unit-variance normalisation applied before training) ---
            # The NN was trained on t_components_norm = (t_components - mean) / std.
            # At inference we must invert this before passing to t_comp_pca.inverse_transform.
            t_comp_scaler_path = self.METADATA_DIR / "t_comp_scaler"
            if t_comp_scaler_path.exists():
                self.t_comp_scaler: Optional[TComponentScaler] = joblib.load(t_comp_scaler_path)
                logging.info("[PkEmulator] t_comp_scaler loaded — will invert NN output normalisation.")
            else:
                # Graceful fallback for models trained before this fix (no-op scaler)
                self.t_comp_scaler = None
                logging.warning(
                    "[PkEmulator] t_comp_scaler not found in metadata directory. "
                    "Assuming model was trained on raw (un-normalised) t-components. "
                    "Re-train with the updated train_utils to apply the 288x std fix."
                )

            # --- Neural network ---
            model_file = self.MODEL_DIR / _model_filename(cosmo_type, prior_type, nl_type)
            logging.info(f"[PkEmulator] Loading model: {model_file}")
            self.model = keras.models.load_model(
                model_file,
                custom_objects={
                    "CustomActivationLayer": CustomActivationLayer,
                    "mse": MeanSquaredError(),
                    # [NEW] Also register Huber so models saved after the loss change load cleanly
                    "huber_loss": Huber(),
                },
            )

            @tf.function(jit_compile=False)
            def _compiled_inference(x):
                return self.model(x, training=False)
            self._compiled_inference = _compiled_inference

            # --- Per-redshift PCA and scalers ---
            self.PCAS: Dict[float, PCA] = {}
            self.SCALERS: Dict[float, object] = {}
            self._pcas_loaded = False
            self.INVERSE_TRANSFORM_MATRICES: Optional[np.ndarray] = None
            self.INVERSE_TRANSFORM_OFFSETS:  Optional[np.ndarray] = None
            self._load_pcas_and_scalers()

            # --- Warm-up ---
            logging.info("[PkEmulator] Warming up neural network...")
            dummy = self.param_scaler.transform(
                np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, 0.0]], dtype=np.float32)
            )
            dummy_tf = tf.constant(dummy, dtype=tf.float32)
            _ = self._compiled_inference(dummy_tf)
            _ = self._compiled_inference(dummy_tf)

            logging.info("[PkEmulator] Initialisation complete.")

        except FileNotFoundError as e:
            logging.error(f"Required file not found: {e.filename}")
            logging.warning(
                f"Check that '{self.METADATA_DIR}' and '{self.MODEL_DIR}' "
                "contain all required files."
            )
            raise

    # -------------------------------------------------------
    def _load_pcas_and_scalers(self) -> None:
        """Load per-redshift PCA/scaler files and pre-compute inverse transform matrices."""
        if self._pcas_loaded:
            return

        logging.info("[PkEmulator] Loading per-redshift PCA and scaler objects...")

        try:
            for z in self.Z_MODES:
                z_key = float(f"{z:.3f}")
                self.PCAS[z_key]    = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
                self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

            logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")

            inverse_matrices, inverse_offsets = [], []
            for z in self.Z_MODES:
                z_key  = float(f"{z:.3f}")
                pca    = self.PCAS[z_key]
                scaler = self.SCALERS[z_key]

                if hasattr(scaler, "scale_"):       # sklearn StandardScaler
                    scale = scaler.scale_
                    mean  = scaler.mean_
                elif hasattr(scaler, "std"):         # custom Scaler from train_utils
                    scale = scaler.std
                    mean  = scaler.mean
                else:
                    raise AttributeError(
                        f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
                    )

                inverse_matrices.append(pca.components_ * scale[None, :])  # (N_PCS, N_K_MODES)
                inverse_offsets.append(pca.mean_ * scale + mean)            # (N_K_MODES,)

            self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
            self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

            self._pcas_loaded = True
            logging.info("[PkEmulator] Inverse transformation matrices ready.")

        except FileNotFoundError as e:
            logging.error(f"Required PCA/scaler file not found: {e.filename}")
            raise

    # -------------------------------------------------------
    def _compute_mps_approximation(self, params: np.ndarray) -> np.ndarray:
        """
        Analytical P_lin(k, z) approximation via symbolic_pofk + growth factors.

        Parameters
        ----------
        params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, wa]

        Returns
        -------
        np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
        """
        As, ns, H0_in, Ob, Om, w0, wa = params
        h = H0_in / 100.0

        k_for_plin = self.K_MODES / h
        pk_fid     = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

        a_array = 1.0 / (self.Z_MODES + 1)
        D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)

        growth_factors = (Dz / D0) ** 2 * (Rz / R0)
        result = pk_fid[None, :] * growth_factors[:, None]
        result = result / h**3
        return result.astype(np.float32)

    # -------------------------------------------------------
    def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
        """
        NN inference: normalised params -> log-fractional differences for all z.

        Inference chain
        ---------------
        1. NN(params_norm)         -> t_components_norm   (unit-variance, what the NN learned)
        2. t_comp_scaler.inverse   -> t_components        (raw tPCA space)
        3. t_comp_pca.inverse      -> pcs_flat            (N_ZS * N_PCS)
        4. Fused PCA+scaler invert -> log_frac            (N_ZS, N_K_MODES)

        Parameters
        ----------
        params_norm : 2-D array of shape (1, N_params)

        Returns
        -------
        np.ndarray of shape (N_ZS, N_K_MODES)
        """
        params_tf = tf.constant(params_norm, dtype=tf.float32)

        # Step 1: NN output — normalised t-components
        t_comps_norm = self._compiled_inference(params_tf).numpy()  # (1, N_T_COMPS)

        # Step 2: [NEW] Invert unit-variance normalisation -> raw tPCA space
        if self.t_comp_scaler is not None:
            t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
        else:
            t_comps_raw = t_comps_norm  # fallback: no scaler (old model)

        # Step 3: Inverse tPCA -> flat PCA coefficients
        pcs_flat = self.t_comp_pca.inverse_transform(t_comps_raw).astype(np.float32)

        # Step 4: Reshape (N_ZS, N_PCS) then fused inverse PCA + inverse logfrac scaler
        pcs_z = pcs_flat.reshape(self.N_ZS, self.N_PCS)
        reconstructed = (
            np.einsum("zp,zpk->zk", pcs_z, self.INVERSE_TRANSFORM_MATRICES)
            + self.INVERSE_TRANSFORM_OFFSETS
        )
        return reconstructed.astype(np.float32)

    # -------------------------------------------------------
    def get_pks(
        self,
        params: List[float],
        use_approximation_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return P(k, z) for a single cosmology.

        Parameters
        ----------
        params : list or 1-D array of 7 values [10^9 A_s, ns, H0, Ob, Om, w0, wa].
            For LCDM, pass w0=-1.0 and wa=0.0.
        use_approximation_only : bool, optional
            If True, skip the neural network and return only the symbolic
            P_lin approximation. Default is False (full emulator).

        Returns
        -------
        k_modes : np.ndarray, shape (N_K_MODES,)
        z_modes : np.ndarray, shape (N_ZS,)
        pks     : np.ndarray, shape (N_ZS, N_K_MODES)
        """
        params_array = np.array(params, dtype=np.float32)
        if params_array.ndim != 1 or len(params_array) != 7:
            raise ValueError(
                f"Expected 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa], "
                f"got shape {params_array.shape}. For LCDM pass w0=-1.0, wa=0.0."
            )

        pk_mps = self._compute_mps_approximation(params_array)

        if use_approximation_only:
            return self.K_MODES, self.Z_MODES, pk_mps

        params_norm = self.param_scaler.transform(params_array.reshape(1, -1))
        log_frac    = self._predict_fracs_all_z(params_norm)
        pks         = (np.exp(log_frac) * pk_mps).astype(np.float32)

        return self.K_MODES, self.Z_MODES, pks


# ----------------------------------------------------------------------------------------------------
# Module-level interface with instance caching
# ----------------------------------------------------------------------------------------------------

_emulator_cache: Dict[Tuple[str, str, str], PkEmulator] = {}


def get_emulator(
    cosmo_type: str = "w0wacdm",
    prior_type: str = "constrained",
    nl_type: str = "lin",
    base_model_path: str = "models",
    base_metadata_path: str = "metadata",
    num_batches: int = 15,
) -> PkEmulator:
    """
    Return a (cached) PkEmulator for the requested configuration.

    Repeated calls with the same ``(cosmo_type, prior_type, nl_type)`` return
    the same instance without reloading files.
    """
    if not _DEPENDENCIES_LOADED:
        raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

    cache_key = (cosmo_type, prior_type, nl_type)
    if cache_key in _emulator_cache:
        logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
        return _emulator_cache[cache_key]

    logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
    emulator = PkEmulator(
        cosmo_type=cosmo_type,
        prior_type=prior_type,
        nl_type=nl_type,
        base_model_path=base_model_path,
        base_metadata_path=base_metadata_path,
        num_batches=num_batches,
    )
    _emulator_cache[cache_key] = emulator
    return emulator


def get_pks(
    params: List[float],
    cosmo_type: str = "w0wacdm",
    prior_type: str = "constrained",
    nl_type: str = "lin",
    use_approximation_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function: get P(k, z) without managing emulator instances.

    Parameters
    ----------
    params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, wa]
        For LCDM, pass w0=-1.0 and wa=0.0.
    cosmo_type, prior_type, nl_type : str
        See PkEmulator docstring for valid values.
    use_approximation_only : bool
        If True, skip the NN and return the symbolic approximation only.

    Returns
    -------
    k_modes, z_modes, pks

    Examples
    --------
    # Full emulator
    k, z, pk = get_pks(
        [2.1, 0.965, 68.0, 0.049, 0.31, -0.9, 0.1],
        cosmo_type='w0wacdm', prior_type='constrained', nl_type='mead2020_feedback',
    )

    # Symbolic-only approximation for LCDM
    k, z, pk = get_pks(
        [2.1, 0.965, 68.0, 0.049, 0.31, -1.0, 0.0],
        cosmo_type='lcdm', prior_type='constrained', nl_type='lin',
        use_approximation_only=True,
    )
    """
    emulator = get_emulator(
        cosmo_type=cosmo_type,
        prior_type=prior_type,
        nl_type=nl_type,
    )
    return emulator.get_pks(params, use_approximation_only=use_approximation_only)