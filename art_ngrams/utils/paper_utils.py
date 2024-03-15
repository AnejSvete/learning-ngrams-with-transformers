from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_results_df(
    df: pd.DataFrame,
    grouping_columns: List[str],
    other_columns: List[str],
    final_grouping_columns: List[str],
) -> pd.DataFrame:
    df_mean = (
        df.groupby(grouping_columns + other_columns, dropna=False)
        .mean(["KL", "entropy"])[["KL", "entropy"]]
        .reset_index()
    )
    df_mean = df_mean.rename(columns={"KL": "Mean"})

    df_std = (
        df.groupby(grouping_columns + other_columns, dropna=False)["KL"]
        .std()
        .reset_index()
    )
    df_std = df_std.rename(columns={"KL": "std"})

    df_ = pd.merge(df_mean, df_std, on=grouping_columns + other_columns)

    df_ = (
        df_.groupby(grouping_columns, dropna=False)
        .min(["KL"])[["Mean", "std", "entropy"]]
        .reset_index()
    )

    df__ = (
        df_.groupby(final_grouping_columns, dropna=False)
        .min(["KL"])[["Mean", "std", "entropy"]]
        .reset_index()
    )

    return df__


def heatmaps(
    df: pd.DataFrame,
    axes_of_interest: List[Tuple[str, str]],
    parameter_names: Dict[str, str],
    estimation: str = "mean",
    title: str = "KL",
):
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    mpl.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(
        1, len(axes_of_interest), figsize=(len(axes_of_interest) * 6, 6)
    )
    fig.tight_layout()
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.2
    )

    for jj, (x, y) in enumerate(axes_of_interest):
        data_ = df.groupby([y, x])

        ax = axes[jj] if len(axes_of_interest) > 1 else axes

        if estimation == "mean":
            data_to_plot = data_["Mean"].mean().unstack().sort_index(ascending=False)
        else:  # estimation == "median"
            data_to_plot = data_["Mean"].median().unstack().sort_index(ascending=False)

        mask = data_to_plot.isna()
        sns.heatmap(
            data_to_plot,
            annot=True,
            fmt=".2f",
            ax=ax,
            mask=mask,
            square=True,
            annot_kws={"fontfamily": "Serif"},
        )
        ax.set_title(
            title,
            fontsize=20,
            fontfamily="Serif",
        )
        ax.set_xlabel(
            parameter_names[x] if x in parameter_names else x,
            fontsize=18,
            fontfamily="Serif",
        )
        ax.set_ylabel(
            parameter_names[y] if y in parameter_names else y,
            fontsize=18,
            fontfamily="Serif",
        )
        for tick in ax.get_xticklabels():
            tick.set_fontname("Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Serif")


def trends(  # NOQA: C901
    df: pd.DataFrame,
    parameters: List[str],
    partition_parameters: List[str],
    parameter_names: Dict[str, str],
    fixed_values: Dict[str, Union[str, int, float]] = {},
    x_bins: Dict[str, np.ndarray] = {},
    parameter_colors: Dict[str, Dict[str, str]] = {},
    investigate_parameter: Optional[str] = None,
    quantity_name: str = "KL",
):
    for parameter in parameters:

        filter = np.asarray([True] * df.shape[0])
        for p in fixed_values:
            if p != parameter:
                filter = filter & (df[p] == fixed_values[p])

        _df = df[filter]

        for parameter_ in partition_parameters:
            if parameter == parameter_:
                continue

            if investigate_parameter is not None:
                if _df[investigate_parameter].nunique() == 0:
                    continue

                _, axes = plt.subplots(
                    1,
                    _df[investigate_parameter].nunique(),
                    figsize=(4 * _df[investigate_parameter].nunique(), 4),
                )
            else:
                _, axes = plt.subplots(1, 1, figsize=(4, 4))

            if investigate_parameter is not None:
                columns = sorted(_df[investigate_parameter].unique())
            else:
                columns = [None]

            for jj, column_value in enumerate(columns):
                ax = axes[jj] if investigate_parameter is not None else axes

                if column_value is not None:
                    __df = _df[_df[investigate_parameter] == column_value]
                else:
                    __df = _df

                if __df.shape[0] == 0:
                    continue

                xb = x_bins.get(parameter, __df[parameter].unique())

                sns.regplot(
                    ax=ax,
                    data=__df,
                    x=parameter,
                    y="Mean",
                    x_estimator=np.mean,
                    order=2 if parameter in ["dn", "n_hat"] else 1,
                    color="tab:red",
                    label="All",
                    x_bins=xb,
                )

                handle_boundaries = [len(ax.get_lines())]

                if parameter_ not in __df.columns:
                    continue

                for parameter_value in __df[parameter_].unique():
                    ___df = __df[__df[parameter_] == parameter_value]

                    sns.regplot(
                        ax=ax,
                        data=___df,
                        x=parameter,
                        y="Mean",
                        x_estimator=np.mean,
                        order=2 if parameter in ["dn", "n_hat"] else 1,
                        color=parameter_colors[parameter_][parameter_value],
                        label=parameter_value,
                        x_bins=xb,
                    )

                    handle_boundaries.append(len(ax.get_lines()))

                if parameter not in x_bins:
                    ax.set_xticks(xb)
                    ax.set_xticklabels(xb)
                if investigate_parameter is not None:
                    ax.set_title(column_value)
                ax.set_xlabel(parameter_names[parameter])
                ax.set_ylabel(
                    quantity_name,
                    fontfamily="Serif",
                )
                ax.grid(True, axis="y")

                handles = ax.get_lines()
                handles = [handles[ii - 1] for ii in handle_boundaries]
                ax.legend(
                    handles=handles,
                    labels=["All"] + list(_df[parameter_].unique()),
                    title=parameter_names[parameter_],
                )

            suptitle = f"Trend of {quantity_name} against {parameter_names[parameter]}"
            if parameter_ in parameter_names:
                suptitle += f" with different values of {parameter_names[parameter_]}"

            plt.suptitle(suptitle, fontfamily="Serif")

            plt.tight_layout()
            plt.show()
