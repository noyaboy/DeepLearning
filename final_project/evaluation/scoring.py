# Competition scoring and validation metrics.

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CompetitionScorer:
    def __init__(self, label_scaler=None):
        self.label_scaler = label_scaler

    def compute_score(self, model, val_loader, device="cuda"):
        if not TORCH_AVAILABLE:
            raise ImportError("torch required")
        if self.label_scaler is None:
            raise ValueError("label_scaler required")
        score, _, _, _ = self.compute_validation_score(model, val_loader, self.label_scaler, device)
        return score

    @staticmethod
    def score_phase1(y_true, y_pred, error_bars):
        var = error_bars ** 2
        chi2 = (y_true - y_pred) ** 2 / var
        penalty = 1000 * (y_true - y_pred) ** 2
        return float(-np.sum(chi2 + np.log(var) + penalty, axis=1).mean())

    @staticmethod
    def compute_validation_score(model, val_loader, label_scaler, device="cuda"):
        if not TORCH_AVAILABLE:
            raise ImportError("torch required")

        model.eval()
        all_preds_mean = []
        all_preds_std = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label']

                outputs = model({'image': images})
                mean = outputs['mean'].cpu().numpy()
                std = torch.exp(0.5 * outputs['log_var']).cpu().numpy()

                all_preds_mean.append(mean)
                all_preds_std.append(std)
                all_labels.append(labels.numpy())

        y_pred = np.concatenate(all_preds_mean, axis=0)
        error_bars = np.concatenate(all_preds_std, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

        y_pred = label_scaler.inverse_transform(y_pred)
        y_true = label_scaler.inverse_transform(y_true)
        error_bars = error_bars * label_scaler.scale_

        score = CompetitionScorer.score_phase1(y_true, y_pred, error_bars)
        return score, y_true, y_pred, error_bars

    @staticmethod
    def compute_mse(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    @staticmethod
    def compute_mae(y_true, y_pred):
        return np.abs(y_true - y_pred).mean()

    @staticmethod
    def compute_detailed_diagnostics(y_true, y_pred, error_bars):
        variances = error_bars ** 2
        squared_errors = (y_true - y_pred) ** 2

        chi_squared = squared_errors / variances
        nll_normalized_term = chi_squared.sum(axis=1).mean()

        log_variances = np.log(variances)
        log_var_term = log_variances.sum(axis=1).mean()

        λ = 1000.0
        penalty_term = λ * squared_errors.sum(axis=1).mean()

        score = -(nll_normalized_term + log_var_term + penalty_term)

        chi2_om = chi_squared[:, 0].mean()
        chi2_s8 = chi_squared[:, 1].mean()
        chi2_per_dim_mean = (chi2_om + chi2_s8) / 2.0

        residuals = np.abs(y_true - y_pred)
        within_1sigma = residuals <= error_bars
        cov68_om = within_1sigma[:, 0].mean()
        cov68_s8 = within_1sigma[:, 1].mean()

        within_2sigma = residuals <= (2 * error_bars)
        cov95_om = within_2sigma[:, 0].mean()
        cov95_s8 = within_2sigma[:, 1].mean()

        rmse_om = np.sqrt(squared_errors[:, 0].mean())
        rmse_s8 = np.sqrt(squared_errors[:, 1].mean())

        mean_sigma_om = error_bars[:, 0].mean()
        mean_sigma_s8 = error_bars[:, 1].mean()

        rmse_over_sigma_om = rmse_om / (mean_sigma_om + 1e-10)
        rmse_over_sigma_s8 = rmse_s8 / (mean_sigma_s8 + 1e-10)

        ratio_penalty_over_chi2 = penalty_term / (nll_normalized_term + 1e-10)

        alpha_star = np.sqrt(chi2_per_dim_mean)
        sigma_scale_applied = 1.0

        alpha_temp_om = np.sqrt(chi2_om)
        alpha_temp_s8 = np.sqrt(chi2_s8)

        mse_om = squared_errors[:, 0].mean()
        mse_s8 = squared_errors[:, 1].mean()

        metrics = dict(
            val_competition_score=score, nll_normalized_term=nll_normalized_term,
            log_var_term=log_var_term, penalty_term=penalty_term,
            chi2_om=chi2_om, chi2_s8=chi2_s8, chi2_per_dim_mean=chi2_per_dim_mean,
            cov68_om=cov68_om, cov68_s8=cov68_s8, cov95_om=cov95_om, cov95_s8=cov95_s8,
            rmse_om=rmse_om, rmse_s8=rmse_s8, mean_sigma_om=mean_sigma_om,
            mean_sigma_s8=mean_sigma_s8, rmse_over_sigma_om=rmse_over_sigma_om,
            rmse_over_sigma_s8=rmse_over_sigma_s8, ratio_penalty_over_chi2=ratio_penalty_over_chi2,
            alpha_star=alpha_star, sigma_scale_applied=sigma_scale_applied,
            alpha_temp_om=alpha_temp_om, alpha_temp_s8=alpha_temp_s8,
            mse_om=mse_om, mse_s8=mse_s8
        )
        return {k: float(v) for k, v in metrics.items()}

    @staticmethod
    def compute_validation_diagnostics(model, val_loader, label_scaler, device="cuda"):
        if not TORCH_AVAILABLE:
            raise ImportError("torch required")

        model.eval()
        means, stds, labels = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                out = model({'image': batch['image'].to(device)})
                means.append(out['mean'].cpu().numpy())
                stds.append(torch.exp(0.5 * out['log_var']).cpu().numpy())
                labels.append(batch['label'].numpy())

        y_pred = label_scaler.inverse_transform(np.concatenate(means, axis=0))
        y_true = label_scaler.inverse_transform(np.concatenate(labels, axis=0))
        error_bars = np.concatenate(stds, axis=0) * label_scaler.scale_

        return CompetitionScorer.compute_detailed_diagnostics(y_true, y_pred, error_bars)
