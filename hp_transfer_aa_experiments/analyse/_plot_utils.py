import math

from pathlib import Path

import pandas as pd
import proplot
import seaborn as sns

from matplotlib import pyplot as plt


def _save_fig(fig, output_dir, filename):
    fig.savefig(Path(output_dir) / f"{filename}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(Path(output_dir) / f"{filename}.pgf", bbox_inches="tight", dpi=600)
    print(f'Saved to "{filename}.{{pdf,pgf}}"')


def set_general_plot_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    plt.switch_backend("pgf")
    proplot.rc.update(
        {
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "font.family": "serif",
            "font.serif": [],
            "font.sans-serif": [],
            "font.monospace": [],
            "font.size": "10.00",
            "text.labelsize": "medium",
            "text.titlesize": "medium",
            "bottomlabel.weight": "normal",
            "toplabel.weight": "normal",
            "leftlabel.weight": "normal",
            "tick.labelweight": "normal",
            "title.weight": "normal",
            "pgf.preamble": r"""
                \usepackage{times}
                \usepackage[utf8x]{inputenc}
                \usepackage[T1]{fontenc}
                \usepackage{microtype}
            """,
        }
    )


def get_approach_spelling(approach):
    if approach == "tpe":
        return "TPE"
    elif approach == "tpe2":
        return "TPE2"
    elif approach == "transfer_tpe":
        return "T2PE"
    elif approach == "gp":
        return "GP"
    else:
        raise ValueError


def get_benchmark_spelling(benchmark, adjustment):
    if benchmark == "fcnet_aa":
        benchmark = "FCN"
    elif benchmark == "svm_aa":
        benchmark = "SVM"
    elif benchmark == "xgb_aa":
        benchmark = "XGB"
    elif benchmark == "nas":
        benchmark = "NAS"
    else:
        raise ValueError

    return f"{benchmark}-{adjustment.upper()}"


def get_runtype_spelling(runtype):
    if runtype.startswith("eval_ref"):
        evals = runtype[-2:]
        return f"{evals} Previous Evaluations"
    else:
        raise ValueError


def set_hue_approach_spelling(df):
    df["approach"].replace("tpe", "TPE", inplace=True)
    df["approach"].replace("transfer_tpe", "Best First + Transfer TPE", inplace=True)
    df["approach"].replace("transfer_top", "Only Optimize New", inplace=True)
    df["approach"].replace("transfer_importance", "Drop Unimportant", inplace=True)
    df["approach"].replace("transfer_tpe_no_best_first", "Transfer TPE", inplace=True)
    df["approach"].replace("transfer_tpe_no_ttpe", "Best First", inplace=True)
    return df


def _format(fig, axs, ymajorlocator, yminorlocator, approach_hue=True):
    axs.format(
        ygrid=True,
        ygridminor=True,
        gridlinewidth=0.6,
        gridalpha=0.5,
        gridcolor="gray",
        ylocator=ymajorlocator,
        yminorlocator=yminorlocator,
        ytickminorsize=0,
        xtickminor=False,
    )
    if approach_hue:
        try:
            # Remove individual legends and draw one legend
            for artist in axs.artists:
                artist[0].remove()
            handles, labels = axs.get_legend_handles_labels()[0]
            fig.legend(handles=handles, labels=labels, frame=False, title=False, loc="b")
        except IndexError:
            pass


def _plot_violins(
    ax,
    benchmark_means,
    title,
    ylabel,
    xlabel,
    yline=None,
    approach_hue=False,
    geometric_mean=False,
):
    def draw_quartiles(self, ax, data, support, density, center, split=False):
        mean_ = data.prod() ** (1 / len(data)) if geometric_mean else data.mean()
        self.draw_to_density(
            ax,
            center,
            mean_,
            support,
            density,
            split,
            linewidth=self.linewidth,
            # dashes=[self.linewidth * 3] * 2
        )

    # pylint: disable=protected-access
    sns.categorical._ViolinPlotter.draw_quartiles = draw_quartiles
    # pylint: enable=protected-access

    if yline is not None:
        ax.axhline(y=yline, color="black", linestyle="--", linewidth=0.6)
    sns.violinplot(
        ax=ax,
        data=benchmark_means,
        x="variable",
        y="value",
        hue="approach" if approach_hue else None,
        split=approach_hue,
        cut=0,
        inner="quartile",
        bw=0.3,
        scale="width",
        saturation=1,
        linewidth=0.9,
    )
    sns.despine(ax=ax)
    ax.format(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def plot_aggregates(
    data,
    output_dir,
    filename,
    yline,
    ylabel,
    xlabel,
    ymajorlocator=None,
    yminorlocator=None,
    approach_hue=False,
    geometric_mean=True,
    clip_to_zero=True,
):
    set_general_plot_style()

    data = data.reset_index()
    groups = data.groupby(["benchmark", "adjustment"])

    num_plots = len(groups)
    num_plot_per_row = 2
    num_rows = math.ceil(num_plots / num_plot_per_row)

    fig, axs = proplot.subplots(
        figsize=(5.35, 8), nrows=num_rows, ncols=num_plot_per_row, share=3, span=True
    )

    for ((benchmark, adjustment), df), ax in zip(groups, axs):
        df = df.drop(columns=["benchmark", "trajectory", "adjustment"])
        if approach_hue:
            df = set_hue_approach_spelling(df)
            df = pd.melt(df, id_vars=["approach"])
        else:
            df = pd.melt(df)
        _plot_violins(
            ax,
            df,
            get_benchmark_spelling(benchmark, adjustment),
            ylabel,
            xlabel,
            yline,
            approach_hue,
            geometric_mean,
        )

    _format(fig, axs, ymajorlocator, yminorlocator)

    # Proplot can not share limits yet
    x0 = min(bbox.x0 for bbox in axs.viewLim)
    if clip_to_zero:
        y0 = 0
    else:
        y0 = min(bbox.y0 for bbox in axs.viewLim)
    x1 = max(bbox.x1 for bbox in axs.viewLim)
    y1 = max(bbox.y1 for bbox in axs.viewLim) * 1.1
    axs.format(xlim=(x0, x1), ylim=(y0, y1))

    _save_fig(fig, output_dir, filename=filename)


def plot_global_aggregates(
    data,
    output_dir,
    filename,
    yline,
    ylabel,
    xlabel,
    ymajorlocator=None,
    yminorlocator=None,
    approach_hue=False,
    geometric_mean=False,
    clip_to_zero=True,
):
    set_general_plot_style()

    data = data.reset_index()
    groups = data.groupby("runtype")

    num_plots = len(groups)
    num_plot_per_row = 3
    num_rows = math.ceil(num_plots / num_plot_per_row)

    figsize = (5.35, 2.4 if approach_hue else 2)
    fig, axs = proplot.subplots(
        figsize=figsize,
        nrows=num_rows,
        ncols=num_plot_per_row,
        share=3,
        span=True,
    )
    for (runtype, df), ax in zip(groups, axs):
        if approach_hue:
            df = set_hue_approach_spelling(df)
            grouped_df = df.groupby(["benchmark", "adjustment", "approach"])
        else:
            grouped_df = df.groupby(["benchmark", "adjustment"])

        if geometric_mean:
            benchmark_means = grouped_df.apply(lambda x: x.prod().pow(1 / len(x)))
            # mean_mean = benchmark_means.reset_index().set_index(["benchmark", "adjustment"])
            # # mean_mean = mean_mean[mean_mean.approach == "Best First"]
            # mean_mean = mean_mean.prod().pow(1/len(mean_mean))
        else:
            benchmark_means = grouped_df.mean()

        if approach_hue:
            benchmark_means = benchmark_means.reset_index().set_index(
                ["benchmark", "adjustment"]
            )
            benchmark_means = pd.melt(benchmark_means, id_vars=["approach"])
        else:
            benchmark_means = pd.melt(benchmark_means)

        _plot_violins(
            ax,
            benchmark_means,
            get_runtype_spelling(runtype),
            ylabel,
            xlabel,
            yline,
            approach_hue,
            geometric_mean,
        )

    _format(fig, axs, ymajorlocator, yminorlocator, approach_hue)
    if clip_to_zero:
        axs.format(ylim=(0, axs.viewLim[0].y1))
    _save_fig(fig, output_dir, filename=filename)
