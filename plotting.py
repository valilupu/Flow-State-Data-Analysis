import matplotlib.pyplot as plt
import seaborn as sns


def plotting(df, file_path):
    # CREATE GRAPH WITH POINT DIFFERENCE INDICATING FLOW SPIKES

    plot = sns.relplot(
        data=df, kind="line",
        x="time", y="points_diff",
    )
    plot.add_legend(title=file_path)

    plt.show()

    print(df.describe())