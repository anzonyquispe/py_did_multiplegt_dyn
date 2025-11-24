from .did_multiplegt_main import did_multiplegt_main
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from ._utils import *

class DidMultiplegtDyn:
    def __init__(self, 
        df,
        outcome,
        group,
        time,
        treatment,
        cluster=None,
        effects=1,
        placebo=1,
        normalized=False,
        effects_equal=False,
        controls=None,
        trends_nonparam=None,
        trends_lin=False,
        continuous=0,
        weight=None,
        predict_het=None,
        same_switchers=False,
        same_switchers_pl=False,
        switchers="",
        only_never_switchers=False,
        ci_level=95,
        save_results=None,
        less_conservative_se=False,
        dont_drop_larger_lower=False,
        drop_if_d_miss_before_first_switch=False
    ):

        #### Getting the initial conditions
        self.args = dict(
            df=df,
            outcome=outcome,
            group=group,
            time=time,
            treatment=treatment,
            cluster=cluster,
            effects=effects,
            placebo=placebo,
            normalized=normalized,
            effects_equal=effects_equal,
            controls=controls,
            trends_nonparam=trends_nonparam,
            trends_lin=trends_lin,
            continuous=continuous,
            weight=weight,
            predict_het=predict_het,
            same_switchers=same_switchers,
            same_switchers_pl=same_switchers_pl,
            switchers=switchers,
            only_never_switchers=only_never_switchers,
            ci_level=ci_level,
            save_results=save_results,
            less_conservative_se=less_conservative_se,
            dont_drop_larger_lower=dont_drop_larger_lower,
            drop_if_d_miss_before_first_switch=drop_if_d_miss_before_first_switch
        )

        validated = validate_inputs( **self.args )
    
    def fit( self ):
        ret = did_multiplegt_main( **self.args )
        self.result = ret
        return self

    def summary(self):
        """
        Collect Effects, ATE, and Placebos from result['did_multiplegt_dyn'],
        append them into one DataFrame, and print it nicely.

        Parameters
        ----------
        result : dict-like
            Object containing result['did_multiplegt_dyn']['Effects'],
            ['ATE'], and ['Placebos'].
        hide_ate_nans : bool, default True
            If True, NaNs in the ATE block are replaced by empty strings
            so they are not printed as 'nan'.

        Returns
        -------
        summary_df : pandas.DataFrame
            Combined table with a 'Block' column indicating which piece
            (Effects / ATE / Placebos) each row comes from.
        """
        import pandas as pd

        hide_ate_nans=True
        dyn = self.result["did_multiplegt_dyn"]
        effects  = dyn["Effects"]
        ate      = dyn["ATE"]
        placebos = dyn["Placebos"]

        def _to_df(obj, block_name):
            """Coerce obj to DataFrame and add a 'Block' column."""
            if isinstance(obj, pd.DataFrame):
                df = obj.copy()
            elif isinstance(obj, pd.Series):
                # make it a single-row DataFrame
                df = obj.to_frame().T
            else:
                # scalar or other type
                df = pd.DataFrame({"value": [obj]})
            df = df.reset_index().rename(columns = {'index' : 'Block'})
            return df

        eff_df = _to_df(effects,  "Effects")
        ate_df = _to_df(ate,      "ATE")
        pl_df  = _to_df(placebos, "Placebos")

        # hide NaNs in the ATE block only
        if hide_ate_nans:
            ate_df = ate_df.where(~ate_df.isna(), "")

        # append in any order you prefer; here: Effects, ATE, Placebos
        summary_df = pd.concat([eff_df, ate_df, pl_df], ignore_index=True, sort=False)

        # pretty print without index
        print(summary_df.to_string(index=False))

        return summary_df



    def plot(self, 
        *,
        n_placebos=None,          # number of pre-treatment periods to show (closest to 0)
        n_effects=None,           # number of post-treatment periods to show
        x_label="Time from Treatment",
        y_label="Estimate",
        title=None,
        note=None,
        label_last_post_as_plus=True,
        fit_pretrend_line=False,
        report_pretrend_in_note=False,
        rotate_by_pretrend=False,
        pretrend_line_kwargs=None,
        figsize=(6.5, 4.5),
        pretrend_decimals=3
    ):
        """
        Plot an event-study figure (placebos + effects) using the did_multiplegt_dyn
        result object.

        Parameters
        ----------
        result : dict-like
            Object containing result['did_multiplegt_dyn']['Effects'] and
            result['did_multiplegt_dyn']['Placebos'].

        n_placebos : int or None
            How many pre-treatment periods (placebos) to display. If None, show all.
            If k > 0, shows the k periods closest to 0 (e.g., -k,...,-1).
            If 0, no pre-periods are shown.

        n_effects : int or None
            How many post-treatment periods (effects) to display. If None, show all.
            If k > 0, shows 1,...,k.
            If 0, no post-periods are shown.

        x_label, y_label : str
            Axis labels.

        title : str or None
            Title for the figure.

        note : str or None
            Text at the bottom (e.g., "Notes: ...").

        label_last_post_as_plus : bool
            If True and there are positive event times, the largest positive
            x-tick label is shown as "k+" instead of "k".

        fit_pretrend_line : bool
            If True AND rotate_by_pretrend is False, fit a line on displayed
            pre-treatment coefficients, constrained to pass through (0,0),
            and draw it.

        report_pretrend_in_note : bool
            If True and a pretrend slope can be estimated, append
            "Pre-trend slope = ..." to the note.

        rotate_by_pretrend : bool
            If True, subtract the predicted value from the pretrend line
            from all displayed coefficients (both pre and post) and also
            shift the CI bounds by the same amount. Uses the slope estimated
            from the displayed pre-periods.
            If rotate_by_pretrend is True, fit_pretrend_line is ignored
            (the pretrend would be zero after rotation).

        pretrend_line_kwargs : dict or None
            Extra kwargs to pass to ax.plot for the pretrend line.

        figsize : tuple
            Matplotlib figure size.

        pretrend_decimals : int
            Number of decimals for reporting the pretrend slope in the note.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        result = self.result
        col_est = "Estimate"
        col_lb = "LB CI"
        col_ub = "UB CI"
        time_col = "time"

        # --- extract tables ---
        effects = result["did_multiplegt_dyn"]["Effects"].copy()
        placebos = result["did_multiplegt_dyn"]["Placebos"].copy()

        # --- construct time columns ---
        n_pl_all = placebos.shape[0]
        placebos[time_col] = np.arange(1, n_pl_all+1)*-1
        n_eff_all = effects.shape[0]
        effects[time_col] = np.arange(1, n_eff_all + 1)

        # sort by time
        placebos = placebos.sort_values(time_col)
        effects = effects.sort_values(time_col)

        # --- subset by requested number of placebos/effects ---
        pl = placebos.copy()
        eff = effects.copy()

        if n_placebos is not None:
            if n_placebos > 0:
                pl = pl[pl[time_col] < 0].iloc[-n_placebos:]
            else:
                pl = pl.iloc[0:0]  # empty

        if n_effects is not None:
            if n_effects > 0:
                eff = eff[eff[time_col] > 0].iloc[:n_effects]
            else:
                eff = eff.iloc[0:0]

        # recompute counts after subsetting
        n_pl = pl.shape[0]
        n_eff = eff.shape[0]

        # --- estimate pretrend slope (on displayed pre-periods) ---
        beta_pretrend = None
        if n_pl > 0:
            t_pre = pl[time_col].to_numpy().astype(float)
            y_pre = pl[col_est].to_numpy().astype(float)
            mask = ~np.isnan(t_pre) & ~np.isnan(y_pre)
            t_pre = t_pre[mask]
            y_pre = y_pre[mask]
            denom = np.sum(t_pre ** 2)
            if t_pre.size > 0 and denom > 0:
                beta_pretrend = float(np.sum(t_pre * y_pre) / denom)

        # --- rotate coefficients by pretrend, if requested ---
        if rotate_by_pretrend:
            if beta_pretrend is None:
                raise ValueError(
                    "Cannot rotate by pretrend: no valid pre-periods to estimate the slope."
                )
            for df in (pl, eff):
                if df.shape[0] == 0:
                    continue
                t = df[time_col].to_numpy().astype(float)
                pred = beta_pretrend * t
                df[col_est] = df[col_est] - pred
                df[col_lb] = df[col_lb] - pred
                df[col_ub] = df[col_ub] - pred
            # After rotation, it doesn't make sense to plot the original sloped line
            fit_pretrend_line = False

        # --- style (similar to journal/event-study figure) ---
        sns.set_theme(style="white", context="paper")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12

        fig, ax = plt.subplots(figsize=figsize)

        # --- pre-treatment (placebos): vertical CI + dots ---
        if n_pl > 0:
            ax.vlines(
                x=pl[time_col],
                ymin=pl[col_lb],
                ymax=pl[col_ub],
                color="black",
                linewidth=2,
            )
            ax.scatter(
                pl[time_col],
                pl[col_est],
                color="black",
                s=35,
                zorder=3,
            )

        # --- post-treatment (effects): vertical CI + dots ---
        if n_eff > 0:
            ax.vlines(
                x=eff[time_col],
                ymin=eff[col_lb],
                ymax=eff[col_ub],
                color="black",
                linewidth=2,
            )
            ax.scatter(
                eff[time_col],
                eff[col_est],
                color="black",
                s=35,
                zorder=3,
            )

        # --- always show effect at 0 with CI [0,0] ---
        ax.vlines(0, 0, 0, color="black", linewidth=2)
        ax.scatter(0, 0, color="black", s=35, zorder=4)

        # --- pretrend line (optional, unrotated coordinates) ---
        if fit_pretrend_line and (beta_pretrend is not None):
            # Determine x range just from displayed periods plus 0
            x_candidates = [0]
            if n_pl > 0:
                x_candidates.append(int(pl[time_col].min()))
            if n_eff > 0:
                x_candidates.append(int(eff[time_col].max()))
            x_min = min(x_candidates)
            x_max = max(x_candidates)

            x_line = np.linspace(x_min, x_max, 200)
            y_line = beta_pretrend * x_line

            if pretrend_line_kwargs is None:
                pretrend_line_kwargs = {}
            pretrend_default = {
                "linestyle": "--",
                "linewidth": 1.2,
                "color": "black",
                "alpha": 0.7,
            }
            pretrend_default.update(pretrend_line_kwargs)
            ax.plot(x_line, y_line, **pretrend_default)

        # --- zero lines (axes) ---
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        # --- ticks ---
        x_candidates = [0]
        if n_pl > 0:
            x_candidates.append(int(pl[time_col].min()))
        if n_eff > 0:
            x_candidates.append(int(eff[time_col].max()))
        x_min = min(x_candidates)
        x_max = max(x_candidates)

        xticks = list(range(x_min, x_max + 1))
        xtick_labels = [str(x) for x in xticks]

        if label_last_post_as_plus and x_max > 0:
            idx = xticks.index(x_max)
            xtick_labels[idx] = f"{x_max}"

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        # only horizontal dashed grid
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="0.8")
        ax.xaxis.grid(False)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="both", direction="out", length=4)

        # labels
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label)

        # --- build note (possibly add pretrend info) ---
        final_note = note
        if report_pretrend_in_note and (beta_pretrend is not None):
            slope_txt = f"Pre-trend slope (based on displayed pre-treatment coefficients) = {beta_pretrend:.{pretrend_decimals}f}."
            if final_note is None:
                final_note = slope_txt
            else:
                final_note = final_note.rstrip() + " " + slope_txt

        # title
        if title is not None:
            ax.set_title(title, loc="center", pad=15)

        # note at bottom
        if final_note is not None:
            fig.text(0.5, 0.02, final_note, ha="center", va="bottom", fontsize=9)
            fig.subplots_adjust(top=0.82, bottom=0.20, left=0.10, right=0.97)
        else:
            fig.tight_layout()

        return self
