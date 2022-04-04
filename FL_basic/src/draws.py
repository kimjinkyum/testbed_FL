import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def file_read(file_name):
    data = pd.DataFrame()
    with open(file_name, 'r') as f:
        logs = f.readlines()
        for log in logs:
            tmp = log.rstrip("\n").split("_")
            data = data.append(pd.Series([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])]),
                               ignore_index=True)
    data.columns = ["Class", "Rounds", "Accuracy", "Time"]
    return data


def file_read_class(file_name, class_num):
    data = pd.DataFrame()

    with open(file_name, 'r') as f:
        logs = f.readlines()
        for log in logs:
            tmp = log.rstrip("\n").split("_")
            data = data.append(pd.Series([class_num, float(tmp[1]), float(tmp[2])]), ignore_index=True)
    data.columns = ["Class", "Rounds", "Accuracy"]
    return data


def file_read_KLD(file_name):
    data = pd.DataFrame()

    with open(file_name, 'r') as f:
        logs = f.readlines()
        for log in logs:
            tmp = log.rstrip("\n").split("_")
            # print(float(tmp[0]))
            data = data.append(pd.Series([(float(tmp[0])), float(tmp[1]), float(tmp[2])]), ignore_index=True)
    data.columns = ["KLD", "Rounds", "Accuracy"]
    return data


def file_read_rounds(file_name, round_drop):
    data = pd.DataFrame()

    with open(file_name, 'r') as f:
        logs = f.readlines()
        for log in logs:
            tmp = log.rstrip("\n").split("_")
            data = data.append(pd.Series([round_drop, float(tmp[0]), float(tmp[1]), float(tmp[2])]), ignore_index=True)
    data.columns = ["Round_drop", "Drop", "Rounds", "Accuracy"]
    return data


def draw_plot(data, x, y, hue, r_d=None):
    kld_list = [0, 0.0785, 0.1513, 0.2206, 0.2913, 0.3727, 0.4342, 0.5275]
    fig, ax = plt.subplots(figsize=(15, 5))
    for size in np.unique(data[hue]):
        # tmp = data2[data2[hue] == size]["Accuracy"]
        if hue == "KLD" and size not in kld_list:
            continue
        group_size = data[data[hue] == size].groupby([x]).mean()
        if hue == "KLD":
            label = "KLD value : " + str(size)
        else:
            label = "After " + str(r_d) + "rounds Drop rate :" + str(round(1 - (size), 2))

        ax.plot(np.arange(0, 50), group_size[y], label=label)
        # ax.text(50 + size, group_size["Accuracy"][49], round(group_size["Accuracy"][49], 4))
    if x == "Rounds":
        x_label = "Communication Round"
    else:
        x_label = x
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y)
    # plt.savefig("result/0304/KLD")
    plt.show()


if __name__ == "__main__":
    data = pd.DataFrame()

    # data = file_read_KLD("log/9.log")

    # KLD
    # draw_plot(data, "Rounds", "Accuracy", "KLD")

    stop_round = [10, 25, 40]

    for r in stop_round:
        file_name = "log/0304/Stop_round50_200_" + str(r) + "NonIID_10 (copy).log "
        tmp = file_read_rounds(file_name, r)
        print(len(tmp)/250)
        data = pd.concat([tmp, data])

    print(data)
    draw_plot(data[data.Round_drop == 10], "Rounds", "Accuracy", "Drop", r_d=0.2)
    # draw_plot(data, "Rounds", "Accuracy", hue)
    """
    for i in range(1, 2):
        file_name = "log/acc_gap_round" + str((i+1)*50) + "_200_reduce_0 (copy)" + ".log "

        tmp = file_read(file_name)

        data = pd.concat([tmp, data])
    """
    # round_drop = [10, 25, 40]
    """
    for i in round_drop:
        file_name = "log/acc_gap_round50_200_" + str(i) +"IID (copy).log "
        tmp = file_read_rounds(file_name, i)
        data = pd.concat([tmp, data])
    """
    # data = file_read("log/acc_gap_round100_200_non_IID_10 (copy).log ")
    # data = data[data["Rounds"] < 80]
    # data = file_read_KLD("log/9 (copy).log")
    # print(data)
    """
    # data = data[data["Drop"] == 0.8]
    for r_d in round_drop:
        fig, ax = plt.subplots(figsize=(10, 5))
        data2 = data[data["Round_drop"] == r_d]
        for size in np.unique(data["Drop"]):
            tmp = data2[data2["Drop"] == size]["Accuracy"]
            group_size = data2[data2["Drop"] == size].groupby(["Rounds"]).mean()
            # print(group_size["Accuracy"])
            if size > 10:
                label = "IID"
            else:
                label = "After " + str(r_d) + "rounds Drop rate :" + str(round(1 - (size), 2))
                # label = "KLD value : " + str(size)
            # ax.plot(group_size["Time"], group_size["Accuracy"], label=label)
            ax.plot(np.arange(0, 50), group_size["Accuracy"], label=label)
            print(group_size["Accuracy"][49])
            # ax.text(50 + size, group_size["Accuracy"][49], round(group_size["Accuracy"][49], 4))
        plt.legend()
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy")
        # plt.savefig("result/0304/KLD")
        plt.show()

    # ax.plot(np.arange(0, 50), benchmark_mean["Accuracy"], label="IID")
    # plt.xlim(25)
    # plt.ylim(0.56, 0.57)

    # fig, ax = plt.subplots(figsize=(10, 5))
    tmp = data[data.Rounds == 49]
    sns.set_theme(style="whitegrid")

    ax2 = sns.boxplot(x="Drop", y="Accuracy", hue="Round_drop", data=tmp, palette="Set2", linewidth=1.5)

    # ax2 = tmp.boxplot(grid=False, column=["Accuracy"], by=["Drop", "Round_drop"], rot=45, fontsize="xx-small",return_type='axes')
    # ax2.grid(False)
    # ax2.right_ax.grid(False)
    plt.title('')
    plt.xlabel("Drop_rate")
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], [0.8, 0.6, 0.4, 0.2, 0])
    plt.ylabel("Accuracy")
    plt.suptitle('')
    # plt.savefig("result/Accuracy box plot based on number of trained classes")
    plt.show()

    # data = pd.read_csv("result/data_per_0220_250.csv", index_col=0)
    # benchmark = file_read_class("log/per_class_number.log", 12)

    # data = pd.concat([data, benchmark])

    # benchmark = file_read_class("log/per_class_number.log", 12)
    # benchmark_mean = benchmark.groupby(["Rounds"]).mean()

    # for distibuted_class in range(2, 12, 2):
    # tmp = file_read_class("log/per_class_number_10_500_" + str(distibuted_class) + ".log", distibuted_class)
    # data = pd.concat([data, tmp])

 
    fig, ax = plt.subplots(figsize=(10, 5))
    for size in np.unique(data["Class"]):
        tmp = data[data["Class"] == size]["Accuracy"]
        group_size = data[data["Class"] == size].groupby(["Rounds"]).mean()
        # print(np.arange(1, 50), group_size["Accuracy"])
        if size > 10:
            label = "IID"
        else:
            label = "Number of trained classes:" + str(size)
        ax.plot(np.arange(0, 50), group_size["Accuracy"], label=label)
    # ax.plot(np.arange(0, 50), benchmark_mean["Accuracy"], label="IID")
    plt.legend()
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.savefig("result/Accuracy based on number of trained classes")
    plt.show()
"""
