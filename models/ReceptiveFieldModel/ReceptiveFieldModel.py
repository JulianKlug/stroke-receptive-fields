import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn import linear_model
from .utils import get_undersample_selector_array
from .receptiveField import reshape_to_receptive_field


class ReceptiveFieldModel(BaseEstimator, TransformerMixin):
    """ A model for the prediction of the final infarct in acute ischemic stroke
            from perfusion maps (mostly Tmax) in Tmax

    Ref: Julian Klug, Elisabeth Dirren, Maria Giulia Preti, Paolo Machi, Andreas Kleinschmidt, Maria Isabel Vargas,
    Dimitri Van De Ville, Emmanuel Carrera; Integrating regional perfusion CT information to improve prediction of
    infarction after stroke; JCBFM; ahead of print

    Parameters
    ----------
    receptive_field_dimensions: (rf_x, rf_y, rf_z) receptive field dimensions in the dimensions used
            where rf denotes the distance from the center voxel (this ensures that receptive fields are always centered)
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    See also
    --------
    """

    def __init__(self, receptive_field_dimensions, n_jobs=1, threshold=0.5, verbose=0):
        self.n_jobs = n_jobs
        self.receptive_field_dimensions = receptive_field_dimensions
        self.threshold = threshold
        self.verbose = verbose
        self.model = linear_model.LogisticRegression(verbose=verbose, max_iter=10000, n_jobs = n_jobs)

    def fit(self, X, y, mask=None):
        """
        Fit a receptive field model to the provided data.
        A mask can be applied to regions that should not be include into the training process.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z, n_channels)
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.
        y : ndarray, shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z)
            Output data. 1 - True label / 0 - False label
        mask: ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Mask for training. 1 - included; 2 - excluded region. Optional
        Returns
        -------
        self : object
        """
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        # Get balancing selector --> random subset respecting population wide distribution
        # Balancing chooses only data inside the mask (mask is applied through balancing)
        selected_for_training = get_undersample_selector_array(y, mask, verbose=self.verbose).reshape(-1)

        rf_inputs, rf_outputs = reshape_to_receptive_field(X, y, self.receptive_field_dimensions)

        selected_X, selected_y = rf_inputs[selected_for_training], rf_outputs[selected_for_training]

        self.model = self.model.fit(selected_X, selected_y)
        self.fitted_ = True

        return self

    def predict_proba(self, X):
        """
        Predict the probability of infarction based on the learned parameters.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z, n_channels)
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.
        Returns
        -------
        probas_3D : ndarray, shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z)
            Predicted probability on every image point.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False, allow_nd=True)

        rf_inputs, rf_outputs = reshape_to_receptive_field(X, np.empty(X.shape[0:-1]),
                                                              self.receptive_field_dimensions)

        probas_ = self.model.predict_proba(rf_inputs)
        probas_ = probas_[:, 1] # we are only interested in the proba that a voxel is 1

        # rebuild to original 3D shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z)
        probas_3D = probas_.reshape(X.shape[0:-1])

        return probas_3D

    def transform(self, X, y=None):
        """
        Predict voxel-wise infarction based on the learned parameters.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z, n_channels)
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.
        y : None
        Returns
        -------
        Xt : ndarray, shape (n_samples, n_pixels_x, n_pixels_y, n_pixels_z)
            Output data. 1 - True label / 0 - False label
        """

        probas_3D = self.predict_proba(X)
        Xt = np.full(X.shape[0:-1], 0, dtype=np.float64)
        Xt[probas_3D > self.threshold] = 1

        return Xt
