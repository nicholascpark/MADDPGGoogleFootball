import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def df_graph(df):

    plt.figure(0)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.axhline(x=0)
    plt.title("Evaluation: MADDPG")

def np_graph(x,y,title, xlabel, ylabel, legend, color):

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=0, linestyle= "--", linewidth = 0.5, color = "k")
    plt.title(title)
    plt.grid(linewidth=0.2)
    plt.plot(x, y, label = legend, color = color, linewidth = 0.5)
    plt.legend(fontsize="small")

def moving_average(scores, n=100) :
    ret = np.cumsum(scores, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

if __name__ == '__main__':

    #MADDPG win percentage
    df1 = pd.read_csv('/mnt/maddpg-self-training-index-4fdc9.csv')
    df2 = pd.read_csv('/mnt/ppo1-baseline-training-index-737f2.csv')
    df3 = pd.read_csv('/mnt/ppo2-baseline-training-index-65d05.csv')
    df4 = pd.read_csv('/mnt/ppo3-baseline-training-index-34cb6.csv')

    maddpg_timesteps = df1["timesteps_total"].values
    maddpg_timesteps = np.sort(maddpg_timesteps[np.where(maddpg_timesteps <= 8000000)])
    y1 = df1["custom_metrics/game_result/win_percentage_episode_mean"].values[np.where(maddpg_timesteps <= 8000000)]

    ppo1_timesteps = df2["timesteps_total"].values
    ppo1_timesteps = np.sort(ppo1_timesteps[np.where(ppo1_timesteps <= 8000000)])
    y2 = df2["custom_metrics/game_result/win_percentage_episode_mean"].values[np.where(ppo1_timesteps <= 8000000)]

    ppo2_timesteps = df3["timesteps_total"].values
    ppo2_timesteps = np.sort(ppo2_timesteps[np.where(ppo2_timesteps <= 8000000)])
    y3 = df3["custom_metrics/game_result/win_percentage_episode_mean"].values[np.where(ppo2_timesteps <= 8000000)]

    ppo3_timesteps = df4["timesteps_total"].values
    ppo3_timesteps = np.sort(ppo3_timesteps[np.where(ppo3_timesteps <= 8000000)])
    y4 = df4["custom_metrics/game_result/win_percentage_episode_mean"].values[np.where(ppo3_timesteps <= 8000000)]

    title = "Win Rate Improvement (Training)"
    ylabel = "Win Rate"
    xlabel = "Timesteps Trained"
    color = "g"

    plt.close()
    plt.figure(0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=0, linestyle= "--", linewidth = 0.5, color = "k")
    plt.title(title)
    plt.grid(linewidth=0.2)
    plt.plot(maddpg_timesteps[14:], moving_average(y1, 15),  color = "b", linewidth = 0.7, label="MADDPG",zorder=4)
    plt.plot(ppo1_timesteps[14:], moving_average(y2, 15), color = "g", linewidth = 0.7, label = "PPO1",zorder=3)
    plt.plot(ppo2_timesteps[14:], moving_average(y3, 15), color = "r", linewidth = 0.7, label = "PPO2",zorder=2)
    plt.plot(ppo3_timesteps[14:], moving_average(y4, 15), color = "darkorange", linewidth = 0.7, label = "PPO3",zorder=1)
    plt.xlim(0.5,8.5)
    plt.legend()
    plt.xticks(np.arange(9)*1000000, labels = ["0", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"])
    title = "/mnt/plots/win_rate_improvement_training"
    plt.savefig(title)

    #MADDPG Score reward
    df1 = pd.read_csv('/mnt/maddpg-self-training-index-4fdc9.csv')
    df2 = pd.read_csv('/mnt/ppo1-baseline-training-index-737f2.csv')

    df1 = df1.drop(["custom_metrics/game_result/win_percentage_episode_mean"], axis =1)
    title = "MADDPG Learning Curve (mean eps. rewards)"
    maddpg_timesteps = np.sort(df1["timesteps_total"].values[:12312])
    y1 = df1["custom_metrics/game_result/score_reward_episode_mean"].values[:12312]

    ppo1_timesteps = df2["timesteps_total"].values
    ppo1_timesteps = np.sort(ppo1_timesteps[np.where(ppo1_timesteps <= 8000000)])
    y2 = df2["custom_metrics/game_result/score_reward_episode_mean"].values[np.where(ppo1_timesteps <= 8000000)]
    ylabel = "Mean Episode Rewards"
    xlabel = "Timesteps Trained"

    plt.close()
    plt.figure(1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=0, linestyle= "--", linewidth = 0.5, color = "k")
    plt.title(title)
    plt.grid(linewidth=0.2)
    plt.plot(maddpg_timesteps, y1, color = "b", linewidth = 0.3, label="MADDPG")
    # plt.plot(ppo1_timesteps, y2, color = "r", linewidth = 0.3, label="PPO Baseline 1")
    plt.legend(fontsize="small")
    # plt.ylim(-0.4, 0.3)
    plt.xticks(np.arange(9)*1000000, labels = ["0", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"])
    title = "/mnt/plots/training_score_rewards_comparison"
    plt.savefig(title)

    #Evaluation: TOtal reward per eps comparison
    title = "Total Rewards per Episode (Testing)"
    ylabel = "Total Rewards"
    xlabel = "Episode"

    maddpg_trpe = np.load("/mnt/rldm/scripts/maddpg_total_reward_per_eps.npy")
    # print(maddpg_trpe)

    ppo1_trpe = np.load("/mnt/rldm/scripts/ppo1_rewards_total.npy")
    ppo2_trpe = np.load("/mnt/rldm/scripts/ppo2_rewards_total.npy")
    ppo3_trpe = np.load("/mnt/rldm/scripts/ppo3_rewards_total.npy")


    plt.close()
    plt.figure(3)
    plt.grid(linewidth=0.2)
    plt.title(title)
    plt.plot(np.cumsum(maddpg_trpe)[:100], linewidth=1, color="r", label="MADDPG")
    # plt.plot(np.cumsum(ppo1_trpe)[:100],  linewidth = 1, color = "g", label = "PPO baseline 1")
    # plt.plot(np.cumsum(ppo2_trpe)[:100], linewidth = 1, color = "b", label = "PPO baseline 2")
    # plt.plot(np.cumsum(ppo3_trpe)[:100], linewidth = 1, color = "y", label = "PPO baseline 3")
    plt.legend(fontsize="small")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=0, color = "k", linewidth = 0.5)
    title = "/mnt/plots/" + title
    plt.savefig(title)

    # Evaluation: TOtal Number of Pass attempts  per eps comparison

    title = "Pass Attempts per Episode (Testing)"
    ylabel = "Number of Pass Attempts"
    xlabel = "Episode"

    maddpg_pa = np.load("/mnt/rldm/scripts/maddpg_pass_attempts.npy")
    ppo1_pa = np.load("/mnt/rldm/scripts/ppo1_pass_attempts.npy")
    ppo2_pa = np.load("/mnt/rldm/scripts/ppo2_pass_attempts.npy")
    ppo3_pa = np.load("/mnt/rldm/scripts/ppo3_pass_attempts.npy")

    plt.close()
    plt.figure(4)
    plt.grid(linewidth=0.2)
    plt.title(title)
    plt.plot(np.cumsum(maddpg_pa)[:100], color="b", label="MADDPG")
    plt.plot(np.cumsum(ppo1_pa)[:100],  color = "g", label = "PPO baseline 1")
    plt.plot(np.cumsum(ppo2_pa)[:100],  color = "r", label = "PPO baseline 2")
    plt.plot(np.cumsum(ppo3_pa)[:100],  color = "darkorange", label = "PPO baseline 3")
    plt.yscale("log")
    plt.axhline(y=0, color = "k", linewidth = 0.5)
    plt.legend(fontsize="small")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = "/mnt/plots/" + title
    plt.savefig(title)

    # Evaluation: TOtal Number of Shoot attempts per eps comparison

    title = "Shoot attempts per Episode (Testing)"
    ylabel = "Number of Shoot Attempts"
    xlabel = "Episode"

    maddpg_sa = np.load("/mnt/rldm/scripts/maddpg_shoot_attempts.npy")
    ppo1_sa = np.load("/mnt/rldm/scripts/ppo1_shoot_attempts.npy")
    ppo2_sa = np.load("/mnt/rldm/scripts/ppo2_shoot_attempts.npy")
    ppo3_sa = np.load("/mnt/rldm/scripts/ppo3_shoot_attempts.npy")

    plt.close()
    plt.figure(4)
    plt.grid(linewidth=0.2)
    plt.title(title)
    plt.yscale("log")
    plt.plot(np.cumsum(maddpg_sa)[:100], color="b", label="MADDPG")
    plt.plot(np.cumsum(ppo1_sa)[:100],  color="g", label="PPO baseline 1")
    plt.plot(np.cumsum(ppo2_sa)[:100],  color="r", label="PPO baseline 2")
    plt.plot(np.cumsum(ppo3_sa)[:100],  color="darkorange", label="PPO baseline 3")

    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = "/mnt/plots/" + title
    plt.savefig(title)


    #
    #
    # # Evaluation: TOtal Suce per eps comparison
    # title = "Number of Successful Passes per Episode (Testing)"
    # ylabel = "Successful Passes"
    # xlabel = "Episode"
    #
    # maddpg_nsp = np.load("/mnt/rldm/scripts/maddpg_num_success_pass.npy")
    # ppo1_nsp = np.load("/mnt/rldm/scripts/ppo1_num_success_pass.npy")
    # ppo2_nsp = np.load("/mnt/rldm/scripts/ppo2_num_success_pass.npy")
    # ppo3_nsp = np.load("/mnt/rldm/scripts/ppo3_num_success_pass.npy")
    #
    # plt.close()
    # plt.figure(5)
    # plt.grid(linewidth=0.2)
    # plt.title(title)
    # plt.axhline(y=0, color = "k", linewidth = 0.5)
    # plt.plot(np.cumsum(maddpg_nsp)[:100],  linewidth=0.5, color="b", label="MADDPG")
    # plt.plot(np.cumsum(ppo1_nsp)[:100], linewidth = 0.5, color = "r", label = "PPO baseline 1")
    # plt.plot(np.cumsum(ppo2_nsp)[:100], linewidth = 0.5, color = "g", label = "PPO baseline 2")
    # plt.plot(np.cumsum(ppo3_nsp)[:100],  linewidth = 0.5, color = "darkorange", label = "PPO baseline 3")
    # plt.legend()
    #
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # title = "/mnt/plots/" + title
    # plt.savefig(title)


    #bar graph of win rate

    plt.close()
    plt.figure(6)
    title = "Comparison of Win Rates and Pass Success Rates"

    maddpg_mv = np.load("/mnt/rldm/scripts/maddpg_mean_values.npy")
    maddpg_mv[1] =  maddpg_mv[2] / maddpg_mv[1]
    maddpg_mv = np.around(maddpg_mv[:2], decimals = 5)

    ppo1_mv = np.load("/mnt/rldm/scripts/ppo1_mean_stats.npy")
    ppo1_mv[1] =  ppo1_mv[2] / ppo1_mv[1]
    ppo1_mv = np.around(ppo1_mv[:2], decimals = 5)

    ppo2_mv = np.load("/mnt/rldm/scripts/ppo2_mean_stats.npy")
    ppo2_mv[1] =  ppo2_mv[2] / ppo2_mv[1]
    ppo2_mv = np.around(ppo2_mv[:2], decimals = 5)

    ppo3_mv = np.load("/mnt/rldm/scripts/ppo3_mean_stats.npy")
    ppo3_mv[1] =  ppo3_mv[2] / ppo3_mv[1]
    ppo3_mv = np.around(ppo3_mv[:2], decimals = 5)


    N = 2
    ind = np.arange(N)
    width = 0.2
    plt.grid(linewidth=0.2)

    xvals = maddpg_mv * 100
    bar1 = plt.bar(ind, xvals, 0.17, color='b', label = "MADDPG", edgecolor = "black")
    yvals = ppo1_mv* 100
    bar2 = plt.bar(ind + width, yvals, 0.17, label = "PPO1", color='g',  edgecolor = "black")
    zvals = ppo2_mv* 100
    bar3 = plt.bar(ind + width * 2, zvals, 0.17, color='r', label = "PPO2", edgecolor = "black")
    wvals = ppo3_mv* 100
    bar4 = plt.bar(ind + width * 3, wvals, 0.17, color='darkorange', label = "PPO3",  edgecolor = "black")

    plt.ylabel('Percentage (%)')
    plt.title("MADDPG vs. PPO: Win Rate and Pass Success Rate")

    plt.xticks(ind + width, ['Win Rate', 'Pass Success Rate'])
    plt.yticks([0,10,20,30,40,45,50], [0,10,20,30,40,"...",100])
    plt.legend((bar1, bar2, bar3, bar4), ('MADDPG', 'PPO1', 'PPO2', 'PPO3'))
    # plt.tight_layout()
    plt.bar_label(bar1, padding = 3)
    plt.bar_label(bar2, padding = 3)
    plt.bar_label(bar3, padding = 3)
    plt.bar_label(bar4, padding = 3)
    plt.ylim(-1.1, 50)
    plt.axhline(y=0, color="k", linestyle="--")

    title = "/mnt/plots/" + title
    plt.savefig(title)



    # shoot-to-pass ratio

    title = "Comparison of Shoot-to-pass Ratio"

    maddpg_mv = np.load("/mnt/rldm/scripts/maddpg_mean_values.npy")
    maddpg_mv = np.around(maddpg_mv[3] / maddpg_mv[2], decimals = 5)
    ppo1_mv = np.load("/mnt/rldm/scripts/ppo1_mean_stats.npy")
    ppo1_mv = np.around(ppo1_mv[3] / ppo1_mv[2], decimals = 5)
    ppo2_mv = np.load("/mnt/rldm/scripts/ppo2_mean_stats.npy")
    ppo2_mv = np.around(ppo2_mv[3] / ppo2_mv[2], decimals = 5)
    ppo3_mv = np.load("/mnt/rldm/scripts/ppo3_mean_stats.npy")
    ppo3_mv = np.around(ppo3_mv[3] / ppo3_mv[2], decimals = 5)

    plt.close()
    plt.figure(7)

    N = 1
    ind = np.arange(N)
    width = 0.2
    plt.grid(linewidth=0.2)

    xvals = maddpg_mv * 100
    bar1 = plt.bar(ind, xvals, 0.1, color='b', label = "MADDPG", edgecolor = "black")
    yvals = ppo1_mv* 100
    bar2 = plt.bar(ind + width, yvals, 0.1, label = "PPO1", color='g',  edgecolor = "black")
    zvals = ppo2_mv* 100
    bar3 = plt.bar(ind + width * 2, zvals, 0.1, color='r', label = "PPO2", edgecolor = "black")
    wvals = ppo3_mv* 100
    bar4 = plt.bar(ind + width * 3, wvals, 0.1, color='darkorange', label = "PPO3",  edgecolor = "black")

    plt.ylabel("Shoot-to-Pass ratio")
    plt.title("MADDPG vs. PPO: Shoot-to-Pass Ratio")

    plt.xticks([0,0.2,0.4,0.6], ['MADDPG', 'PPO1', 'PPO2', 'PPO3'])
    plt.legend((bar1, bar2, bar3, bar4), ('MADDPG', 'PPO1', 'PPO2', 'PPO3'))
    plt.ylim(-50,3700)
    plt.bar_label(bar1, padding = 3)
    plt.bar_label(bar2, padding = 3)
    plt.bar_label(bar3, padding = 3)
    plt.bar_label(bar4, padding = 3)
    plt.axhline(y=0, color="k", linestyle="--")
    plt.axhline(y=1, color="k", linestyle="--", linewidth = 0.5)

    title = "/mnt/plots/" + title
    plt.savefig(title)