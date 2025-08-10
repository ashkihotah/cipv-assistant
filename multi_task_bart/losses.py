import torch.nn.functional as F
import torch.nn as nn
import torch

class BartWithRegressionCriterion(nn.Module):
    """
    A custom abstract loss function interface that combines the generation loss from a BART model
    with a regression loss for polarity prediction.
    
    This class is designed to be used with a BART model that outputs both
    generated text and regression predictions.
    """

    def forward(self, model_outputs, true_polarities):
        """
        Computes the combined loss.

        Args:
            model_outputs (dict): Contains 'generation_loss' and 'polarities'.
            true_polarities (torch.Tensor): Ground truth polarities.

        Returns:
            torch.Tensor: The computed loss.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class UncertaintyLoss(BartWithRegressionCriterion):
    """
    Implements the Uncertainty-based loss function from Kendall et al. (2018)
    for multi-task learning.

    This loss function learns to balance multiple task losses by weighting them
    based on their homoscedastic uncertainty. It does this by introducing
    trainable parameters (log_vars) for each task.

    The total loss is calculated as:
    L_total = Σ [ (1 / (2*σ_i²)) * L_i + log(σ_i) ]
    where σ_i is the uncertainty for task i, and L_i is its raw loss.

    To maintain numerical stability, we work with log(σ_i²) instead of σ_i.
    Let s_i = log(σ_i²), then σ_i² = exp(s_i). The formula becomes:
    L_total = Σ [ exp(-s_i) * L_i + 0.5 * s_i ]
    
    We have two tasks: 
    - Task 1: Polarity prediction (Regression)
    - Task 2: Explanation generation (Classification-like via Cross-Entropy)
    """
    def __init__(self, regression_loss_fn=nn.SmoothL1Loss()):
        super(UncertaintyLoss, self).__init__()
        # Initialize two trainable log-variance parameters, one for each task.
        # We initialize them to 0.0, which means the initial variance σ² is exp(0) = 1.
        # This gives both tasks equal initial weighting.
        # We name them to make their purpose clear.
        self.log_var_polarity = nn.Parameter(torch.zeros(1))
        self.log_var_explanation = nn.Parameter(torch.zeros(1))
        self.regression_loss_fn = regression_loss_fn  # Loss function for regression task

    def forward(self, model_outputs, true_polarities):
        """
        Calculates the combined, uncertainty-weighted loss.

        Args:
            model_outputs (dict): Contains the model's outputs:
                - 'polarities': Predicted polarities (regression output).
                - 'generation_loss': Pre-computed generation loss (e.g., from BART).
            true_polarities (torch.Tensor): Ground truth polarities for regression.

        Returns:
            torch.Tensor: The final, combined loss to be backpropagated.
        """
        predicted_polarities = model_outputs['polarities']
        loss_explanation = model_outputs['generation_loss']  # BART's pre-computed generation loss

        # Compute the Smooth L1 Loss
        loss_polarities = self.regression_loss_fn(predicted_polarities, true_polarities)

        # Ensure the parameters are on the same device as the input losses
        self.log_var_polarity.data = self.log_var_polarity.data.to(loss_polarities.device)
        self.log_var_explanation.data = self.log_var_explanation.data.to(loss_explanation.device)

        # Calculate the precision terms (1 / (2*σ²))
        # The factor of 0.5 for the regression precision term comes from the
        # Gaussian likelihood derivation in the paper.
        precision_polarity = torch.exp(-self.log_var_polarity)
        
        # The precision term for classification-like losses is (1 / σ²)
        precision_explanation = torch.exp(-self.log_var_explanation)

        # The final loss for each task component
        # Regression term: (1/(2σ²))*L_reg + log(σ) = (1/(2σ²))*L_reg + 0.5*log(σ²)
        term_polarity = 0.5 * precision_polarity * loss_polarities + 0.5 * self.log_var_polarity
        
        # Explanation term: For simplicity and stability, many implementations use the same
        # form for all tasks. This is a robust choice.
        # Here we use the general form: (1/σ²)*L_gen + log(σ) = (1/σ²)*L_gen + 0.5*log(σ²)
        # term_explanation = precision_explanation * loss_explanation + 0.5 * self.log_var_explanation
        
        # A more common and stable reparameterization for all tasks is:
        # L_i_final = exp(-s_i) * L_i + s_i
        # Let's use this for the explanation part for robustness.
        term_explanation = precision_explanation * loss_explanation + self.log_var_explanation

        # The total loss is the sum of the individual task losses
        total_loss = term_polarity + term_explanation

        return {
            'total_loss': total_loss,
            'reg_loss': loss_polarities,
            'gen_loss': loss_explanation
        }

class StaticWeightedLoss(BartWithRegressionCriterion):
    """
    A dedicated class to compute the combined loss for the hybrid model.
    This separates the loss logic from the model's forward pass.
    """
    def __init__(self, regression_loss_fn=nn.MSELoss(), alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Weight for regression loss
        self.regression_loss_fn = regression_loss_fn

    def forward(self, model_outputs, true_polarities):
        # Unpack model outputs
        predicted_polarities = model_outputs['polarities']
        gen_loss = model_outputs['generation_loss']  # BART's pre-computed generation loss

        # Calculate Regression Loss (MSE) for each sample in the batch
        reg_loss = self.regression_loss_fn(predicted_polarities, true_polarities)

        total_loss = (self.alpha * reg_loss) + ((1 - self.alpha) * gen_loss)
        return {
            'total_loss': total_loss,
            'reg_loss': reg_loss,
            'gen_loss': gen_loss
        }

class EuclideanLoss(BartWithRegressionCriterion):
    """
    A dedicated class to compute the combined loss for the hybrid model.
    This separates the loss logic from the model's forward pass.
    """
    def __init__(self, regression_loss_fn=nn.MSELoss()):
        super().__init__()
        self.regression_loss_fn = regression_loss_fn

    def forward(self, model_outputs, true_polarities):
        # Unpack model outputs
        predicted_polarities = model_outputs['polarities']
        gen_loss = model_outputs['generation_loss']  # BART's pre-computed generation loss

        # Calculate Regression Loss (MSE) for each sample in the batch
        reg_loss = self.regression_loss_fn(predicted_polarities, true_polarities)

        # Compute the Euclidean norm of the vector [reg_loss, gen_loss] for each sample
        total_loss = torch.sqrt(reg_loss * reg_loss + gen_loss * gen_loss)
        return {
            'total_loss': total_loss,  # Return the mean loss across the batch
            'reg_loss': reg_loss,
            'gen_loss': gen_loss
        }
