from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import copy
sns.set_style('darkgrid')


def diff_horizon_asy():
    df_list = []
    Step_asy_4X7 = n['Step_asy_4X7'].reshape(200,)
    step = []
    for i in range(200):
        step.append(Step_asy_4X7[i] - 10)
    Step_asy_4X5 = n['Step_asy_4X5'].reshape(100, )
    step_low = []
    for i in range(100):
        step_low.append(Step_asy_4X5[i] - 10)
    Value_asy_4X5 = n['Value_asy_4X5'].reshape(200,)
    Value_asy_4X7 = n['Value_asy_4X7'].reshape(200,)
    Value_asy_4X12 = n['Value_asy_4X12'].reshape(200, )
    Value_asy_4X15 = n['Value_asy_4X15'].reshape(200, )
    Value_asy_4X20 = n['Value_asy_4X20'].reshape(200, )
    Value_asy_4X25 = n['Value_asy_4X25'].reshape(100, )
    Value_asy_low = n['y_low'].reshape(200, )

    df_for_asy_4X5 = pd.DataFrame({'Algorithms': 'Asy',
                                'Iteration': step,
                                'Value': Value_asy_4X5,
                                'PH': 'PH: 20'})
    df_for_asy_4X7 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Iteration': step,
                                   'Value': Value_asy_4X7,
                                   'PH': 'PH: 28'})
    df_for_asy_4X12 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Iteration': step,
                                   'Value': Value_asy_4X12,
                                   'PH': 'PH: 48'})
    df_for_asy_4X15 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Iteration': step,
                                   'Value': Value_asy_4X15,
                                   'PH': 'PH: 60'})
    df_for_asy_4X20 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Iteration': step,
                                   'Value': Value_asy_4X20,
                                   'PH': 'PH: 80'})
    df_for_asy_4X25 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Iteration': step_low,
                                   'Value': Value_asy_4X25,
                                   'PH': 'PH: 100'})
    df_for_asy_upper = pd.DataFrame({'Algorithms': 'Asy',
                                   'Iteration': step,
                                   'Value': Value_asy_low,
                                   'PH': 'Reward Target'})
    df_for_this_asy = df_for_asy_4X5.append(df_for_asy_4X7, ignore_index=True)
    df_for_this_asy = df_for_this_asy.append(df_for_asy_4X12, ignore_index=True)
    df_for_this_asy = df_for_this_asy.append(df_for_asy_4X15, ignore_index=True)
    df_for_this_asy = df_for_this_asy.append(df_for_asy_4X20, ignore_index=True)
    df_for_this_asy = df_for_this_asy.append(df_for_asy_4X25, ignore_index=True)
    df_for_this_asy = df_for_this_asy.append(df_for_asy_upper, ignore_index=True)
    df_list.append(df_for_this_asy)
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.80])
    sns.lineplot(x="Iteration", y="Value", hue='PH', style='PH', data=df_list[0],
                 linewidth=2, palette="bright")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4], handles[5], handles[6]], labels=[labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]], loc='lower right', frameon=False, fontsize=20)
    ax1.set_ylabel('Reward', fontsize=25)
    ax1.set_xlabel("Iterations", fontsize=25)
    #ax1.set_title("Asynchronous T3S scheme", loc='center', y=1.0, fontsize=22)
    plt.xlim(0, 1000)
    plt.ylim(-50, 0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()


def diff_horizon_sy():
    df_list = []
    Step_sy_4X5 = n['Step_sy_4X5'].reshape(100,)
    step = []
    for i in range(100):
        step.append(Step_sy_4X5[i] * 10)

    Step_asy_4X7 = n['Step_asy_4X7'].reshape(200, )
    step_low = []
    for i in range(200):
        step_low.append(Step_asy_4X7[i] - 10)

    Value_sy_4X5 = n['Value_sy_4X5'].reshape(100,)
    Value_sy_4X7 = n['Value_sy_4X7'].reshape(100,)
    Value_sy_4X12 = n['Value_sy_4X12'].reshape(100, )
    Value_sy_4X15 = n['Value_sy_4X15'].reshape(100, )
    Value_sy_4X20 = n['Value_sy_4X20'].reshape(100, )
    Value_sy_4X25 = n['Value_sy_4X25'].reshape(100, )
    Value_asy_low = n['y_low'].reshape(200, )
    # print('Step_asy_4X7 = ', Step_asy_4X7)
    df_for_sy_4X5 = pd.DataFrame({'Algorithms': 'Syn',
                                'Iteration': step,
                                'Value': Value_sy_4X5,
                                'PH': 'PH: 20'})

    df_for_sy_4X7 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Iteration': step,
                                   'Value': Value_sy_4X7,
                                   'PH': 'PH: 28'})
    df_for_sy_4X12 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Iteration': step,
                                   'Value': Value_sy_4X12,
                                   'PH': 'PH: 48'})
    df_for_sy_4X15 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Iteration': step,
                                   'Value': Value_sy_4X15,
                                   'PH': 'PH: 60'})
    df_for_sy_4X20 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Iteration': step,
                                   'Value': Value_sy_4X20,
                                   'PH': 'PH: 80'})
    df_for_sy_4X25 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Iteration': step,
                                   'Value': Value_sy_4X25,
                                   'PH': 'PH: 100'})
    df_for_asy_upper = pd.DataFrame({'Algorithms': 'Syn',
                                   'Iteration': step_low,
                                   'Value': Value_asy_low,
                                   'PH': 'Reward Target'})
    df_for_this_sy = df_for_sy_4X5.append(df_for_sy_4X7, ignore_index=True)
    df_for_this_sy = df_for_this_sy.append(df_for_sy_4X12, ignore_index=True)
    df_for_this_sy = df_for_this_sy.append(df_for_sy_4X15, ignore_index=True)
    df_for_this_sy = df_for_this_sy.append(df_for_sy_4X20, ignore_index=True)
    df_for_this_sy = df_for_this_sy.append(df_for_sy_4X25, ignore_index=True)
    df_for_this_sy = df_for_this_sy.append(df_for_asy_upper, ignore_index=True)
    df_list.append(df_for_this_sy)
    print('df_list = ', df_list)
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.80])
    sns.lineplot(x="Iteration", y="Value", hue='PH', style='PH', data=df_list[0],
                 linewidth=2, palette="bright")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4], handles[5], handles[6]], labels=[labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]], loc='lower right', frameon=False, fontsize=20)
    ax1.set_ylabel('Reward', fontsize=25)
    ax1.set_xlabel("Iterations", fontsize=25)
    #ax1.set_title("Synchronous T3S scheme", loc='center', y=1.0, fontsize=22)
    plt.xlim(0, 1000)
    plt.ylim(-50, 0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()


def diff_horizon_adp():
    df_list = []
    Step_ADP_20 = n['Step_ADP_20'].reshape(150, )
    Step_asy_4X7 = n['Step_asy_4X7'].reshape(200, )
    step = []
    for i in range(200):
        step.append(Step_asy_4X7[i] - 10)
    Value_ADP_20 = n['Value_ADP_20'].reshape(150, )
    Value_ADP_28 = n['Value_ADP_28'].reshape(150, )
    Value_ADP_48 = n['Value_ADP_48'].reshape(150, )
    Value_ADP_60 = n['Value_ADP_60'].reshape(150, )
    Value_ADP_80 = n['Value_ADP_80'].reshape(150, )
    Value_ADP_100 = n['Value_ADP_100'].reshape(150, )
    Value_asy_low = n['y_low'].reshape(200, )

    df_for_ADP_20 = pd.DataFrame({'Algorithms': 'ADP',
                                   'Iteration': Step_ADP_20,
                                   'Value': Value_ADP_20,
                                   'PH': 'PH: 20'})
    df_for_ADP_28 = pd.DataFrame({'Algorithms': 'ADP',
                                   'Iteration': Step_ADP_20,
                                   'Value': Value_ADP_28,
                                   'PH': 'PH: 28'})
    df_for_ADP_48 = pd.DataFrame({'Algorithms': 'ADP',
                                    'Iteration': Step_ADP_20,
                                    'Value': Value_ADP_48,
                                    'PH': 'PH: 48'})
    df_for_ADP_60 = pd.DataFrame({'Algorithms': 'ADP',
                                    'Iteration': Step_ADP_20,
                                    'Value': Value_ADP_60,
                                    'PH': 'PH: 60'})
    df_for_ADP_80 = pd.DataFrame({'Algorithms': 'ADP',
                                    'Iteration': Step_ADP_20,
                                    'Value': Value_ADP_80,
                                    'PH': 'PH: 80'})
    df_for_ADP_100 = pd.DataFrame({'Algorithms': 'ADP',
                                    'Iteration': Step_ADP_20,
                                    'Value': Value_ADP_100,
                                    'PH': 'PH: 100'})
    df_for_asy_upper = pd.DataFrame({'Algorithms': 'ADP',
                                     'Iteration': step,
                                     'Value': Value_asy_low,
                                     'PH': 'Reward Target'})
    df_for_this_ADP = df_for_ADP_20.append(df_for_ADP_28, ignore_index=True)
    df_for_this_ADP = df_for_this_ADP.append(df_for_ADP_48, ignore_index=True)
    df_for_this_ADP = df_for_this_ADP.append(df_for_ADP_60, ignore_index=True)
    df_for_this_ADP = df_for_this_ADP.append(df_for_ADP_80, ignore_index=True)
    df_for_this_ADP = df_for_this_ADP.append(df_for_ADP_100, ignore_index=True)
    df_for_this_ADP = df_for_this_ADP.append(df_for_asy_upper, ignore_index=True)
    df_list.append(df_for_this_ADP)
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.80])
    sns.lineplot(x="Iteration", y="Value", hue='PH', style='PH', data=df_list[0],
                 linewidth=2, palette="bright")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4], handles[5], handles[6]],
               labels=[labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]],
               loc='lower right', frameon=False, fontsize=20)
    ax1.set_ylabel('Reward', fontsize=25)
    ax1.set_xlabel("Iterations", fontsize=25)
    #ax1.set_title("MBPG scheme", loc='center', y=1.0, fontsize=22)
    plt.xlim(0, 140)
    plt.ylim(-50, 0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()


def tracking_performance():
    df_list_diff_init_point = []

    df_for_ref_0 = pd.DataFrame({'Algorithms': 'Ref',
                                   'Position_X': m['x_ADP_20'][0],
                                   'Position_Y': m['y_ref'][0],
                                   'PH': 'Reference Trajectory',
                                   'index': '0'})
    # df_for_ref_1 = pd.DataFrame({'Algorithms': 'Ref',
    #                                'Position_X': m['x_ADP_20'][1],
    #                                'Position_Y': m['y_ref'][1],
    #                                'PH': 'Reference Trajectory',
    #                                'index': '1'})
    # df_for_ref_2 = pd.DataFrame({'Algorithms': 'Ref',
    #                              'Position_X': m['x_ADP_20'][2],
    #                              'Position_Y': m['y_ref'][2],
    #                              'PH': 'PH: 20',
    #                              'index': '2'})
    # df_for_ref_3 = pd.DataFrame({'Algorithms': 'Ref',
    #                              'Position_X': m['x_ADP_20'][3],
    #                              'Position_Y': m['y_ref'][3],
    #                              'PH': 'Reference Trajectory',
    #                              'index': '3'})
    # df_for_ref_4 = pd.DataFrame({'Algorithms': 'Ref',
    #                              'Position_X': m['x_ADP_20'][4],
    #                              'Position_Y': m['y_ref'][4],
    #                              'PH': 'Reference Trajectory',
    #                              'index': '4'})

    # prediction = 20
    df_for_ADP_20_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': m['x_ADP_20'][0],
                                   'Position_Y': m['y_ADP_20'][0],
                                   'PH': 'PH: 20',
                                   'index': '0'})
    df_for_ADP_20_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': m['x_ADP_20'][1],
                                   'Position_Y': m['y_ADP_20'][1],
                                   'PH': 'PH: 20',
                                   'index': '1'})
    df_for_ADP_20_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': m['x_ADP_20'][2],
                                 'Position_Y': m['y_ADP_20'][2],
                                 'PH': 'PH: 20',
                                 'index': '2'})
    df_for_ADP_20_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': m['x_ADP_20'][3],
                                 'Position_Y': m['y_ADP_20'][3],
                                 'PH': 'PH: 20',
                                 'index': '3'})
    df_for_ADP_20_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': m['x_ADP_20'][4],
                                 'Position_Y': m['y_ADP_20'][4],
                                 'PH': 'PH: 20',
                                 'index': '4'})

    df_for_syn_20_0 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_sy_20'][0],
                                    'PH': 'PH: 20',
                                    'index': '0'})
    df_for_syn_20_1 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_sy_20'][1],
                                    'PH': 'PH: 20',
                                    'index': '1'})
    df_for_syn_20_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_sy_20'][2],
                                    'PH': 'PH: 20',
                                    'index': '2'})
    df_for_syn_20_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_sy_20'][3],
                                    'PH': 'PH: 20',
                                    'index': '3'})
    df_for_syn_20_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_sy_20'][4],
                                    'PH': 'PH: 20',
                                    'index': '4'})


    df_for_asy_20_0 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_asy_20'][0],
                                    'PH': 'PH: 20',
                                    'index': '0'})
    df_for_asy_20_1 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_asy_20'][1],
                                    'PH': 'PH: 20',
                                    'index': '1'})
    df_for_asy_20_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_asy_20'][2],
                                    'PH': 'PH: 20',
                                    'index': '2'})
    df_for_asy_20_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_asy_20'][3],
                                    'PH': 'PH: 20',
                                    'index': '3'})
    df_for_asy_20_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_asy_20'][4],
                                    'PH': 'PH: 20',
                                    'index': '4'})

    # prediction = 48
    df_for_ADP_48_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_ADP_48'][0],
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_ADP_48_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_ADP_48'][1],
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_ADP_48_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_ADP_48'][2],
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_ADP_48_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_ADP_48'][3],
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_ADP_48_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_ADP_48'][4],
                                    'PH': 'PH: 48',
                                    'index': '4'})

    df_for_syn_48_0 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_sy_48'][0],
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_syn_48_1 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_sy_48'][1],
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_syn_48_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_sy_48'][2],
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_syn_48_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_sy_48'][3],
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_syn_48_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_sy_48'][4],
                                    'PH': 'PH: 48',
                                    'index': '4'})

    df_for_asy_48_0 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_asy_48'][0],
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_asy_48_1 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_asy_48'][1],
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_asy_48_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_asy_48'][2],
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_asy_48_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_asy_48'][3],
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_asy_48_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_asy_48'][4],
                                    'PH': 'PH: 48',
                                    'index': '4'})


    # prediction = 100
    df_for_ADP_100_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][0],
                                     'Position_Y': m['y_ADP_100'][0],
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_ADP_100_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][1],
                                     'Position_Y': m['y_ADP_100'][1],
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_ADP_100_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][2],
                                     'Position_Y': m['y_ADP_100'][2],
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_ADP_100_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][3],
                                     'Position_Y': m['y_ADP_100'][3],
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_ADP_100_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][4],
                                     'Position_Y': m['y_ADP_100'][4],
                                    'PH': 'PH: 100',
                                    'index': '4'})

    df_for_syn_100_0 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][0],
                                     'Position_Y': m['y_sy_100'][0],
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_syn_100_1 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][1],
                                     'Position_Y': m['y_sy_100'][1],
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_syn_100_2 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][2],
                                     'Position_Y': m['y_sy_100'][2],
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_syn_100_3 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][3],
                                     'Position_Y': m['y_sy_100'][3],
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_syn_100_4 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][4],
                                     'Position_Y': m['y_sy_100'][4],
                                    'PH': 'PH: 100',
                                    'index': '4'})

    df_for_asy_100_0 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][0],
                                     'Position_Y': m['y_asy_100'][0],
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_asy_100_1 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][1],
                                     'Position_Y': m['y_asy_100'][1],
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_asy_100_2 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][2],
                                     'Position_Y': m['y_asy_100'][2],
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_asy_100_3 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][3],
                                     'Position_Y': m['y_asy_100'][3],
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_asy_100_4 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][4],
                                     'Position_Y': m['y_asy_100'][4],
                                    'PH': 'PH: 100',
                                    'index': '4'})


    df_for_0 = df_for_ref_0.append(df_for_ADP_20_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_ADP_48_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_ADP_100_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_syn_20_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_syn_48_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_syn_100_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_asy_20_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_asy_48_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_asy_100_0, ignore_index=True)


    df_for_1 = df_for_ref_0.append(df_for_ADP_20_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_ADP_48_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_ADP_100_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_syn_20_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_syn_48_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_syn_100_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_asy_20_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_asy_48_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_asy_100_1, ignore_index=True)


    df_for_2 = df_for_ref_0.append(df_for_ADP_20_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_ADP_48_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_ADP_100_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_syn_20_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_syn_48_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_syn_100_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_asy_20_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_asy_48_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_asy_100_2, ignore_index=True)


    df_for_3 = df_for_ref_0.append(df_for_ADP_20_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_ADP_48_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_ADP_100_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_syn_20_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_syn_48_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_syn_100_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_asy_20_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_asy_48_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_asy_100_3, ignore_index=True)


    df_for_4 = df_for_ref_0.append(df_for_ADP_20_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_ADP_48_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_ADP_100_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_syn_20_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_syn_48_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_syn_100_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_asy_20_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_asy_48_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_asy_100_4, ignore_index=True)

    df_list_diff_init_point.append(df_for_0)
    df_list_diff_init_point.append(df_for_1)
    df_list_diff_init_point.append(df_for_2)
    df_list_diff_init_point.append(df_for_3)
    df_list_diff_init_point.append(df_for_4)

    # df_for_20 = df_for_ref_0.append(df_for_ADP_20_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_ADP_20_1, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_ADP_20_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_ADP_20_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_ADP_20_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_20_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_20_1, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_20_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_20_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_20_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_20_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_20_1, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_20_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_20_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_20_4, ignore_index=True)
    #
    #
    # df_for_48 = df_for_ref_0.append(df_for_ADP_48_0, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_ADP_48_1, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_ADP_48_2, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_ADP_48_3, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_ADP_48_4, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_syn_48_0, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_syn_48_1, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_syn_48_2, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_syn_48_3, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_syn_48_4, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_asy_48_0, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_asy_48_1, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_asy_48_2, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_asy_48_3, ignore_index=True)
    # df_for_48 = df_for_48.append(df_for_asy_48_4, ignore_index=True)
    #
    #
    # df_for_100 = df_for_ref_0.append(df_for_ADP_100_0, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_ADP_100_1, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_ADP_100_2, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_ADP_100_3, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_ADP_100_4, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_syn_100_0, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_syn_100_1, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_syn_100_2, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_syn_100_3, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_syn_100_4, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_asy_100_0, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_asy_100_1, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_asy_100_2, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_asy_100_3, ignore_index=True)
    # df_for_100 = df_for_100.append(df_for_asy_100_4, ignore_index=True)


    # df_list_diff_prediction = []
    #
    # df_list_diff_prediction.append(df_for_20)
    # df_list_diff_prediction.append(df_for_48)
    # df_list_diff_prediction.append(df_for_100)


    #print(df_list_diff_init_point[0])
    f1 = plt.figure(1)
    # # ax1 = f1.add_axes([0.1, 0.55, 0.35, 0.35])
    # ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    # sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', data=df_list_diff_init_point[0],
    #              linewidth=2, palette="bright")
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=15)
    # ax1.set_ylabel('Lateral Position', fontsize=20)
    # ax1.set_xlabel("Longitudinal Position", fontsize=20)
    # plt.xlim(0, 400)
    # plt.tick_params(labelsize=20)



    # ax2 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    # sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', data=df_list_diff_init_point[2],
    #              linewidth=2, palette="bright")
    # handles, labels = ax2.get_legend_handles_labels()
    # ax2.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=15)
    # ax2.set_ylabel('Lateral Position', fontsize=20)
    # ax2.set_xlabel("Longitudinal Position", fontsize=20)
    # plt.xlim(0, 400)
    # plt.tick_params(labelsize=20)
    #
    #
    # #
    # ax3 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    # sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', data=df_list_diff_init_point[3],
    #              linewidth=2, palette="bright")
    # handles, labels = ax3.get_legend_handles_labels()
    # ax3.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=15)
    # ax3.set_ylabel('Lateral Position', fontsize=20)
    # ax3.set_xlabel("Longitudinal Position", fontsize=20)
    # plt.xlim(0, 400)
    # plt.tick_params(labelsize=20)
    #
    #
    # #
    ax4 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', data=df_list_diff_init_point[4],
                 linewidth=2, palette="bright")
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=15)
    ax4.set_ylabel('Lateral Position', fontsize=20)
    ax4.set_xlabel("Longitudinal Position", fontsize=20)
    plt.xlim(0, 400)

    plt.tick_params(labelsize=20)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)

    plt.show()


def tracking_error():
    df_list_diff_init_point = []
    n = 200
    Upper_bound = copy.deepcopy(m['x_ADP_20'][0])
    Lower_bound = copy.deepcopy(m['x_ADP_20'][0])
    for i in range(n):
        Upper_bound[i] = 0.4
        Lower_bound[i] = - 0.4
    df_for_upper_bound = pd.DataFrame({'Algorithms': 'Upper Bound',
                                   'Position_X': m['x_ADP_20'][0],
                                   'Position_Y': Upper_bound,
                                   'PH': 'Upper Bound',
                                   'index': '0'})
    df_for_lower_bound = pd.DataFrame({'Algorithms': 'Lower Bound',
                                   'Position_X': m['x_ADP_20'][1],
                                   'Position_Y': Lower_bound,
                                   'PH': 'Lower Bound',
                                   'index': '1'})
    # prediction = 20
    df_for_ADP_20_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': m['x_ADP_20'][0],
                                   'Position_Y': m['y_ADP_20'][0] - m['y_ref'][0],
                                   'PH': 'PH: 20',
                                   'index': '0'})
    df_for_ADP_20_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': m['x_ADP_20'][1],
                                   'Position_Y': m['y_ADP_20'][1] - m['y_ref'][0],
                                   'PH': 'PH: 20',
                                   'index': '1'})
    df_for_ADP_20_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': m['x_ADP_20'][2],
                                 'Position_Y': m['y_ADP_20'][2] - m['y_ref'][0],
                                 'PH': 'PH: 20',
                                 'index': '2'})
    df_for_ADP_20_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': m['x_ADP_20'][3],
                                 'Position_Y': m['y_ADP_20'][3] - m['y_ref'][0],
                                 'PH': 'PH: 20',
                                 'index': '3'})
    df_for_ADP_20_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': m['x_ADP_20'][4],
                                 'Position_Y': m['y_ADP_20'][4] - m['y_ref'][0],
                                 'PH': 'PH: 20',
                                 'index': '4'})

    df_for_syn_20_0 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_sy_20'][0] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '0'})
    df_for_syn_20_1 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_sy_20'][1] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '1'})
    df_for_syn_20_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_sy_20'][2] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '2'})
    df_for_syn_20_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_sy_20'][3] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '3'})
    df_for_syn_20_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_sy_20'][4] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '4'})


    df_for_asy_20_0 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_asy_20'][0] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '0'})
    df_for_asy_20_1 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_asy_20'][1] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '1'})
    df_for_asy_20_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_asy_20'][2] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '2'})
    df_for_asy_20_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_asy_20'][3] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '3'})
    df_for_asy_20_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_asy_20'][4] - m['y_ref'][0],
                                    'PH': 'PH: 20',
                                    'index': '4'})

    # prediction = 48
    df_for_ADP_48_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_ADP_48'][0] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_ADP_48_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_ADP_48'][1] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_ADP_48_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_ADP_48'][2] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_ADP_48_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_ADP_48'][3] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_ADP_48_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_ADP_48'][4] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '4'})

    df_for_syn_48_0 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_sy_48'][0] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_syn_48_1 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_sy_48'][1] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_syn_48_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_sy_48'][2] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_syn_48_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_sy_48'][3] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_syn_48_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_sy_48'][4] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '4'})

    df_for_asy_48_0 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][0],
                                    'Position_Y': m['y_asy_48'][0] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_asy_48_1 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][1],
                                    'Position_Y': m['y_asy_48'][1] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_asy_48_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][2],
                                    'Position_Y': m['y_asy_48'][2] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_asy_48_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][3],
                                    'Position_Y': m['y_asy_48'][3] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_asy_48_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': m['x_ADP_20'][4],
                                    'Position_Y': m['y_asy_48'][4] - m['y_ref'][0],
                                    'PH': 'PH: 48',
                                    'index': '4'})


    # prediction = 100
    df_for_ADP_100_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][0],
                                     'Position_Y': m['y_ADP_100'][0] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_ADP_100_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][1],
                                     'Position_Y': m['y_ADP_100'][1] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_ADP_100_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][2],
                                     'Position_Y': m['y_ADP_100'][2] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_ADP_100_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][3],
                                     'Position_Y': m['y_ADP_100'][3] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_ADP_100_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': m['x_ADP_20'][4],
                                     'Position_Y': m['y_ADP_100'][4] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '4'})

    df_for_syn_100_0 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][0],
                                     'Position_Y': m['y_sy_100'][0] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_syn_100_1 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][1],
                                     'Position_Y': m['y_sy_100'][1] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_syn_100_2 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][2],
                                     'Position_Y': m['y_sy_100'][2] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_syn_100_3 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][3],
                                     'Position_Y': m['y_sy_100'][3] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_syn_100_4 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': m['x_ADP_20'][4],
                                     'Position_Y': m['y_sy_100'][4] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '4'})

    df_for_asy_100_0 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][0],
                                     'Position_Y': m['y_asy_100'][0] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_asy_100_1 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][1],
                                     'Position_Y': m['y_asy_100'][1] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_asy_100_2 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][2],
                                     'Position_Y': m['y_asy_100'][2] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_asy_100_3 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][3],
                                     'Position_Y': m['y_asy_100'][3] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_asy_100_4 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': m['x_ADP_20'][4],
                                     'Position_Y': m['y_asy_100'][4] - m['y_ref'][0],
                                    'PH': 'PH: 100',
                                    'index': '4'})


    df_for_0 = df_for_upper_bound.append(df_for_lower_bound, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_ADP_20_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_ADP_48_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_ADP_100_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_syn_20_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_syn_48_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_syn_100_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_asy_20_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_asy_48_0, ignore_index=True)
    df_for_0 = df_for_0.append(df_for_asy_100_0, ignore_index=True)


    df_for_1 = df_for_upper_bound.append(df_for_lower_bound, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_ADP_20_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_ADP_48_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_ADP_100_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_syn_20_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_syn_48_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_syn_100_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_asy_20_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_asy_48_1, ignore_index=True)
    df_for_1 = df_for_1.append(df_for_asy_100_1, ignore_index=True)


    df_for_2 = df_for_upper_bound.append(df_for_lower_bound, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_ADP_20_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_ADP_48_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_ADP_100_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_syn_20_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_syn_48_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_syn_100_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_asy_20_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_asy_48_2, ignore_index=True)
    df_for_2 = df_for_2.append(df_for_asy_100_2, ignore_index=True)


    df_for_3 = df_for_upper_bound.append(df_for_lower_bound, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_ADP_20_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_ADP_48_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_ADP_100_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_syn_20_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_syn_48_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_syn_100_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_asy_20_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_asy_48_3, ignore_index=True)
    df_for_3 = df_for_3.append(df_for_asy_100_3, ignore_index=True)


    df_for_4 = df_for_upper_bound.append(df_for_lower_bound, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_ADP_20_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_ADP_48_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_ADP_100_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_syn_20_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_syn_48_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_syn_100_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_asy_20_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_asy_48_4, ignore_index=True)
    df_for_4 = df_for_4.append(df_for_asy_100_4, ignore_index=True)

    df_list_diff_init_point.append(df_for_0)
    df_list_diff_init_point.append(df_for_1)
    df_list_diff_init_point.append(df_for_2)
    df_list_diff_init_point.append(df_for_3)
    df_list_diff_init_point.append(df_for_4)


    f1 = plt.figure(1)
    # ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    # sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', style='Algorithms', data=df_list_diff_init_point[0],
    #              linewidth=2, palette="bright")
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4]], labels=[labels[0], labels[1], labels[2], labels[3], 'Asy'], loc='upper right', frameon=False, fontsize=15)
    # ax1.set_ylabel('Lateral Position', fontsize=20)
    # ax1.set_xlabel("Longitudinal Position", fontsize=20)
    # plt.xlim(0, 400)
    # plt.tick_params(labelsize=20)


    # ax2 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    # sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', style='Algorithms', data=df_list_diff_init_point[2],
    #              linewidth=2, palette="bright")
    # handles, labels = ax2.get_legend_handles_labels()
    # ax2.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4]], labels=[labels[0], labels[1], labels[2], labels[3], 'Asy'], loc='lower right', frameon=False, fontsize=15)
    # ax2.set_ylabel('Lateral Position', fontsize=20)
    # ax2.set_xlabel("Longitudinal Position", fontsize=20)
    # plt.xlim(0, 400)
    # plt.ylim(-1.5, 0.5)
    # plt.tick_params(labelsize=20)
    #
    #
    # ax3 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    # sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', style='Algorithms', data=df_list_diff_init_point[3],
    #              linewidth=2, palette="bright")
    # handles, labels = ax3.get_legend_handles_labels()
    # ax3.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4]], labels=[labels[0], labels[1], labels[2], labels[3], 'Asy'], loc='lower right', frameon=False, fontsize=15)
    # ax3.set_ylabel('Lateral Position', fontsize=20)
    # ax3.set_xlabel("Longitudinal Position", fontsize=20)
    # plt.xlim(0, 400)
    # plt.tick_params(labelsize=20)
    #
    #
    ax4 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', style='Algorithms', data=df_list_diff_init_point[4],
                 linewidth=2, palette="bright", hue_order=['Upper Bound', 'Lower Bound', 'MBPG', 'Syn', 'Asy'])
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4],], labels=[labels[0], labels[1], labels[2], labels[3], 'Asy'], loc='lower right', frameon=False, fontsize=15)
    ax4.set_ylabel('Lateral Position', fontsize=20)
    ax4.set_xlabel("Longitudinal Position", fontsize=20)
    plt.xlim(0, 400)
    plt.tick_params(labelsize=20)


    plt.show()


def adp_value_reward():
    adp_value['Step_asy'] = adp_value['Step_asy'].reshape(200, )
    adp_value['y_low'] = adp_value['y_low'].reshape(200, )
    adp_value['Step_ADP'] = adp_value['Step_ADP'].reshape(150, )
    # print(adp_value['Value_adp_20_1'].reshape(150, ))
    # exit()
    df_for_ref_0 = pd.DataFrame({'Algorithms': 'Ref',
                                   'Position_X': adp_value['Step_asy'],
                                   'Position_Y': adp_value['y_low'],
                                   'PH': 'Reward Target',
                                   'index': '0'})
    # print('df_for_ref_0 = ', df_for_ref_0)

    # prediction = 20
    df_for_ADP_20_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_20_1'].reshape(150, ),
                                   'PH': 'PH: 20',
                                   'index': '0'})
    df_for_ADP_20_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_20_2'].reshape(150, ),
                                   'PH': 'PH: 20',
                                   'index': '1'})
    df_for_ADP_20_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_20_3'].reshape(150, ),
                                 'PH': 'PH: 20',
                                 'index': '2'})
    df_for_ADP_20_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_20_4'].reshape(150, ),
                                 'PH': 'PH: 20',
                                 'index': '3'})
    df_for_ADP_20_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': adp_value['Step_ADP'],
                                 'Position_Y': adp_value['Value_adp_20_5'].reshape(150, ),
                                 'PH': 'PH: 20',
                                 'index': '4'})

    # prediction = 28
    df_for_ADP_28_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_28_1'].reshape(150, ),
                                    'PH': 'PH: 28',
                                    'index': '0'})
    df_for_ADP_28_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_28_2'].reshape(150, ),
                                    'PH': 'PH: 28',
                                    'index': '1'})
    df_for_ADP_28_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_28_3'].reshape(150, ),
                                    'PH': 'PH: 28',
                                    'index': '2'})
    df_for_ADP_28_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_28_4'].reshape(150, ),
                                    'PH': 'PH: 28',
                                    'index': '3'})
    df_for_ADP_28_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_28_5'].reshape(150, ),
                                    'PH': 'PH: 28',
                                    'index': '4'})

    # prediction = 48
    df_for_ADP_48_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_48_1'].reshape(150, ),
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_ADP_48_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_48_2'].reshape(150, ),
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_ADP_48_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_48_3'].reshape(150, ),
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_ADP_48_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_48_4'].reshape(150, ),
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_ADP_48_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_48_5'].reshape(150, ),
                                    'PH': 'PH: 48',
                                    'index': '4'})

    # prediction = 60
    df_for_ADP_60_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_60_1'].reshape(150, ),
                                    'PH': 'PH: 60',
                                    'index': '0'})
    df_for_ADP_60_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_60_2'].reshape(150, ),
                                    'PH': 'PH: 60',
                                    'index': '1'})
    df_for_ADP_60_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_60_3'].reshape(150, ),
                                    'PH': 'PH: 60',
                                    'index': '2'})
    df_for_ADP_60_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_60_4'].reshape(150, ),
                                    'PH': 'PH: 60',
                                    'index': '3'})
    df_for_ADP_60_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_60_5'].reshape(150, ),
                                    'PH': 'PH: 60',
                                    'index': '4'})

    # prediction = 80
    df_for_ADP_80_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_80_1'].reshape(150, ),
                                    'PH': 'PH: 80',
                                    'index': '0'})
    df_for_ADP_80_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_80_2'].reshape(150, ),
                                    'PH': 'PH: 80',
                                    'index': '1'})
    df_for_ADP_80_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_80_3'].reshape(150, ),
                                    'PH': 'PH: 80',
                                    'index': '2'})
    df_for_ADP_80_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_80_4'].reshape(150, ),
                                    'PH': 'PH: 80',
                                    'index': '3'})
    df_for_ADP_80_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': adp_value['Step_ADP'],
                                   'Position_Y': adp_value['Value_adp_80_5'].reshape(150, ),
                                    'PH': 'PH: 80',
                                    'index': '4'})

    # prediction = 100
    df_for_ADP_100_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': adp_value['Step_ADP'],
                                     'Position_Y': adp_value['Value_adp_100_1'].reshape(150, ),
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_ADP_100_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': adp_value['Step_ADP'],
                                     'Position_Y': adp_value['Value_adp_100_2'].reshape(150, ),
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_ADP_100_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': adp_value['Step_ADP'],
                                     'Position_Y': adp_value['Value_adp_100_3'].reshape(150, ),
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_ADP_100_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': adp_value['Step_ADP'],
                                     'Position_Y': adp_value['Value_adp_100_4'].reshape(150, ),
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_ADP_100_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': adp_value['Step_ADP'],
                                     'Position_Y': adp_value['Value_adp_100_5'].reshape(150, ),
                                    'PH': 'PH: 100',
                                    'index': '4'})


    df_for_20 = df_for_ref_0.append(df_for_ADP_20_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_20_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_20_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_20_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_20_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_28_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_28_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_28_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_28_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_28_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_48_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_48_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_48_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_48_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_48_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_60_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_60_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_60_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_60_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_60_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_ADP_80_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_80_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_80_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_80_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_80_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_100_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_100_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_100_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_100_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_ADP_100_4, ignore_index=True)

    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_20)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='PH', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    #print('labels = ', labels)
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3], handles[4], handles[5], handles[6]], labels=[labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]], loc='lower right', frameon=False, fontsize=30)
    ax1.set_ylabel('Reward', fontsize=35)
    ax1.set_xlabel("Iteration", fontsize=35)
    plt.xlim(0, 149)
    plt.ylim(-50, 0)
    plt.tick_params(labelsize=30)

    plt.show()


def asy_value_reward():
    asy_value['Step_asy'] = asy_value['Step_asy'].reshape(200, )
    asy_value['y_low'] = asy_value['y_low'].reshape(200, )
    #adp_value['Step_ADP'] = adp_value['Step_ADP'].reshape(150, )
    # print(asy_value['Value_asy_20_1'].reshape(200, ))
    # exit()
    df_for_ref_0 = pd.DataFrame({'Algorithms': 'Ref',
                                   'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['y_low'],
                                   'PH': 'Reward Target',
                                   'index': '0'})
    # print('df_for_ref_0 = ', df_for_ref_0)

    # prediction = 20
    df_for_asy_20_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_20_1'].reshape(200, ),
                                   'PH': 'PH: 20',
                                   'index': '0'})
    df_for_asy_20_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_20_2'].reshape(200, ),
                                   'PH': 'PH: 20',
                                   'index': '1'})
    df_for_asy_20_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_20_3'].reshape(200, ),
                                 'PH': 'PH: 20',
                                 'index': '2'})
    df_for_asy_20_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_20_4'].reshape(200, ),
                                 'PH': 'PH: 20',
                                 'index': '3'})
    df_for_asy_20_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': asy_value['Step_asy'],
                                 'Position_Y': asy_value['Value_asy_20_5'].reshape(200, ),
                                 'PH': 'PH: 20',
                                 'index': '4'})

    # prediction = 28
    df_for_asy_28_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_28_1'].reshape(200, ),
                                    'PH': 'PH: 28',
                                    'index': '0'})
    df_for_asy_28_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_28_2'].reshape(200, ),
                                    'PH': 'PH: 28',
                                    'index': '1'})
    df_for_asy_28_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_28_3'].reshape(200, ),
                                    'PH': 'PH: 28',
                                    'index': '2'})
    df_for_asy_28_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_28_4'].reshape(200, ),
                                    'PH': 'PH: 28',
                                    'index': '3'})
    df_for_asy_28_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_28_5'].reshape(200, ),
                                    'PH': 'PH: 28',
                                    'index': '4'})

    # prediction = 48
    df_for_asy_48_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_48_1'].reshape(200, ),
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_asy_48_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_48_2'].reshape(200, ),
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_asy_48_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_48_3'].reshape(200, ),
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_asy_48_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_48_4'].reshape(200, ),
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_asy_48_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_48_5'].reshape(200, ),
                                    'PH': 'PH: 48',
                                    'index': '4'})

    # prediction = 60
    df_for_asy_60_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_60_1'].reshape(200, ),
                                    'PH': 'PH: 60',
                                    'index': '0'})
    df_for_asy_60_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_60_2'].reshape(200, ),
                                    'PH': 'PH: 60',
                                    'index': '1'})
    df_for_asy_60_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_60_3'].reshape(200, ),
                                    'PH': 'PH: 60',
                                    'index': '2'})
    df_for_asy_60_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_60_4'].reshape(200, ),
                                    'PH': 'PH: 60',
                                    'index': '3'})
    df_for_asy_60_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_60_5'].reshape(200, ),
                                    'PH': 'PH: 60',
                                    'index': '4'})

    # prediction = 80
    df_for_asy_80_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_80_1'].reshape(200, ),
                                    'PH': 'PH: 80',
                                    'index': '0'})
    df_for_asy_80_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_80_2'].reshape(200, ),
                                    'PH': 'PH: 80',
                                    'index': '1'})
    df_for_asy_80_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_80_3'].reshape(200, ),
                                    'PH': 'PH: 80',
                                    'index': '2'})
    df_for_asy_80_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_80_4'].reshape(200, ),
                                    'PH': 'PH: 80',
                                    'index': '3'})
    df_for_asy_80_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': asy_value['Step_asy'],
                                   'Position_Y': asy_value['Value_asy_80_5'].reshape(200, ),
                                    'PH': 'PH: 80',
                                    'index': '4'})

    # prediction = 100
    df_for_asy_100_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': asy_value['Step_asy'],
                                     'Position_Y': asy_value['Value_asy_100_1'].reshape(200, ),
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_asy_100_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': asy_value['Step_asy'],
                                     'Position_Y': asy_value['Value_asy_100_2'].reshape(200, ),
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_asy_100_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': asy_value['Step_asy'],
                                     'Position_Y': asy_value['Value_asy_100_3'].reshape(200, ),
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_asy_100_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': asy_value['Step_asy'],
                                     'Position_Y': asy_value['Value_asy_100_4'].reshape(200, ),
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_asy_100_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': asy_value['Step_asy'],
                                     'Position_Y': asy_value['Value_asy_100_5'].reshape(200, ),
                                    'PH': 'PH: 100',
                                    'index': '4'})


    df_for_20 = df_for_ref_0.append(df_for_asy_20_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_48_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_4, ignore_index=True)

    # # df_for_20 = df_for_20.append(df_for_asy_60_1, ignore_index=True)
    # df_for_20 = df_for_ref_0.append(df_for_asy_60_1, ignore_index=True)
    # # df_for_20 = df_for_20.append(df_for_asy_60_0, ignore_index=True)
    # # df_for_20 = df_for_20.append(df_for_asy_60_2, ignore_index=True)
    # # df_for_20 = df_for_20.append(df_for_asy_60_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_60_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_80_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_80_1, ignore_index=True)
    # # df_for_20 = df_for_20.append(df_for_asy_80_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_80_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_80_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_100_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_100_1, ignore_index=True)
    # # df_for_20 = df_for_20.append(df_for_asy_100_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_100_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_100_4, ignore_index=True)

    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_20)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='PH', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=30)
    ax1.set_ylabel('Reward', fontsize=35)
    ax1.set_xlabel("Iteration", fontsize=35)
    plt.xlim(0, 1200)
    plt.ylim(-50, 0)
    plt.tick_params(labelsize=30)

    plt.show()


def syn_value_reward():
    syn_value['Step_asy'] = syn_value['Step_asy'].reshape(200, )
    syn_value['Step_syn'] = syn_value['Step_syn'].reshape(100, )
    syn_value['y_low'] = syn_value['y_low'].reshape(200, )
    #adp_value['Step_ADP'] = adp_value['Step_ADP'].reshape(150, )
    # print(asy_value['Value_asy_20_1'].reshape(200, ))
    # exit()
    df_for_ref_0 = pd.DataFrame({'Algorithms': 'Ref',
                                   'Position_X': syn_value['Step_asy'],
                                   'Position_Y': syn_value['y_low'],
                                   'PH': 'Reward Target',
                                   'index': '0'})
    # print('df_for_ref_0 = ', df_for_ref_0)

    # prediction = 20
    df_for_syn_20_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_20_1'].reshape(100, ),
                                   'PH': 'PH: 20',
                                   'index': '0'})
    df_for_syn_20_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                   'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_20_2'].reshape(100, ),
                                   'PH': 'PH: 20',
                                   'index': '1'})
    df_for_syn_20_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_20_3'].reshape(100, ),
                                 'PH': 'PH: 20',
                                 'index': '2'})
    df_for_syn_20_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_20_4'].reshape(100, ),
                                 'PH': 'PH: 20',
                                 'index': '3'})
    df_for_syn_20_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'Position_X': syn_value['Step_syn'],
                                 'Position_Y': syn_value['Value_syn_20_5'].reshape(100, ),
                                 'PH': 'PH: 20',
                                 'index': '4'})

    # prediction = 28
    df_for_syn_28_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_28_1'].reshape(100, ),
                                    'PH': 'PH: 28',
                                    'index': '0'})
    df_for_syn_28_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_28_2'].reshape(100, ),
                                    'PH': 'PH: 28',
                                    'index': '1'})
    df_for_syn_28_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_28_3'].reshape(100, ),
                                    'PH': 'PH: 28',
                                    'index': '2'})
    df_for_syn_28_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_28_4'].reshape(100, ),
                                    'PH': 'PH: 28',
                                    'index': '3'})
    df_for_syn_28_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_28_5'].reshape(100, ),
                                    'PH': 'PH: 28',
                                    'index': '4'})

    # prediction = 48
    df_for_syn_48_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_48_1'].reshape(100, ),
                                    'PH': 'PH: 48',
                                    'index': '0'})
    df_for_syn_48_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_48_2'].reshape(100, ),
                                    'PH': 'PH: 48',
                                    'index': '1'})
    df_for_syn_48_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_48_3'].reshape(100, ),
                                    'PH': 'PH: 48',
                                    'index': '2'})
    df_for_syn_48_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_48_4'].reshape(100, ),
                                    'PH': 'PH: 48',
                                    'index': '3'})
    df_for_syn_48_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_48_5'].reshape(100, ),
                                    'PH': 'PH: 48',
                                    'index': '4'})

    # prediction = 60
    df_for_syn_60_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_60_1'].reshape(100, ),
                                    'PH': 'PH: 60',
                                    'index': '0'})
    df_for_syn_60_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_60_2'].reshape(100, ),
                                    'PH': 'PH: 60',
                                    'index': '1'})
    df_for_syn_60_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_60_3'].reshape(100, ),
                                    'PH': 'PH: 60',
                                    'index': '2'})
    df_for_syn_60_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_60_4'].reshape(100, ),
                                    'PH': 'PH: 60',
                                    'index': '3'})
    df_for_syn_60_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_60_5'].reshape(100, ),
                                    'PH': 'PH: 60',
                                    'index': '4'})

    # prediction = 80
    df_for_syn_80_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_80_1'].reshape(100, ),
                                    'PH': 'PH: 80',
                                    'index': '0'})
    df_for_syn_80_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_80_2'].reshape(100, ),
                                    'PH': 'PH: 80',
                                    'index': '1'})
    df_for_syn_80_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_80_3'].reshape(100, ),
                                    'PH': 'PH: 80',
                                    'index': '2'})
    df_for_syn_80_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_80_4'].reshape(100, ),
                                    'PH': 'PH: 80',
                                    'index': '3'})
    df_for_syn_80_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                    'Position_X': syn_value['Step_syn'],
                                   'Position_Y': syn_value['Value_syn_80_5'].reshape(100, ),
                                    'PH': 'PH: 80',
                                    'index': '4'})

    # prediction = 100
    df_for_syn_100_0 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': syn_value['Step_syn'],
                                     'Position_Y': syn_value['Value_syn_100_1'].reshape(100, ),
                                    'PH': 'PH: 100',
                                    'index': '0'})
    df_for_syn_100_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': syn_value['Step_syn'],
                                     'Position_Y': syn_value['Value_syn_100_2'].reshape(100, ),
                                    'PH': 'PH: 100',
                                    'index': '1'})
    df_for_syn_100_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': syn_value['Step_syn'],
                                     'Position_Y': syn_value['Value_syn_100_3'].reshape(100, ),
                                    'PH': 'PH: 100',
                                    'index': '2'})
    df_for_syn_100_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': syn_value['Step_syn'],
                                     'Position_Y': syn_value['Value_syn_100_4'].reshape(100, ),
                                    'PH': 'PH: 100',
                                    'index': '3'})
    df_for_syn_100_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': syn_value['Step_syn'],
                                     'Position_Y': syn_value['Value_syn_100_5'].reshape(100, ),
                                    'PH': 'PH: 100',
                                    'index': '4'})


    df_for_20 = df_for_ref_0.append(df_for_syn_20_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_48_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_48_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_48_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_48_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_48_4, ignore_index=True)

    # df_for_20 = df_for_ref_0.append(df_for_syn_60_0, ignore_index=True)
    # # df_for_20 = df_for_20.append(df_for_syn_60_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_60_1, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_60_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_60_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_60_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_80_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_80_1, ignore_index=True)
    # #df_for_20 = df_for_20.append(df_for_syn_80_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_80_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_80_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_100_0, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_100_1, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_100_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_100_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_100_4, ignore_index=True)

    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_20)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='PH', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=30)
    ax1.set_ylabel('Reward', fontsize=35)
    ax1.set_xlabel("Iteration", fontsize=35)
    plt.xlim(0, 1200)
    plt.ylim(-50, 0)
    plt.tick_params(labelsize=30)

    plt.show()


def qushi():
    df_for_adp_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': [20],
                                     'Position_Y': [0.26]})
    df_for_adp_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': [28],
                                     'Position_Y': [0.34]})
    df_for_adp_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': [48],
                                     'Position_Y': [0.59]})
    df_for_adp_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': [60],
                                     'Position_Y': [0.68]})
    df_for_adp_5 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': [80],
                                     'Position_Y': [0.97]})
    df_for_adp_6 = pd.DataFrame({'Algorithms': 'MBPG',
                                     'Position_X': [100],
                                     'Position_Y': [1.18]})

    df_for_syn_1 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': [20],
                                     'Position_Y': [0.20]})
    df_for_syn_2 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': [28],
                                     'Position_Y': [0.21]})
    df_for_syn_3 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': [48],
                                     'Position_Y': [0.23]})
    df_for_syn_4 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': [60],
                                     'Position_Y': [0.22]})
    df_for_syn_5 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': [80],
                                     'Position_Y': [0.22]})
    df_for_syn_6 = pd.DataFrame({'Algorithms': 'Syn',
                                     'Position_X': [100],
                                     'Position_Y': [0.25]})

    df_for_asy_1 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': [20],
                                     'Position_Y': [0.056]})
    df_for_asy_2 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': [28],
                                     'Position_Y': [0.058]})
    df_for_asy_3 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': [48],
                                     'Position_Y': [0.061]})
    df_for_asy_4 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': [60],
                                     'Position_Y': [0.063]})
    df_for_asy_5 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': [80],
                                     'Position_Y': [0.066]})
    df_for_asy_6 = pd.DataFrame({'Algorithms': 'Asy',
                                     'Position_X': [100],
                                     'Position_Y': [0.071]})


    df_for_adp = df_for_adp_1.append(df_for_adp_2, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_adp_3, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_adp_4, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_adp_5, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_adp_6, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_syn_1, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_syn_2, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_syn_3, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_syn_4, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_syn_5, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_syn_6, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_asy_1, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_asy_2, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_asy_3, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_asy_4, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_asy_5, ignore_index=True)
    df_for_adp = df_for_adp.append(df_for_asy_6, ignore_index=True)
    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_adp)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='Algorithms', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2]], labels=[labels[0], labels[1], labels[2]], loc='upper left', frameon=False, fontsize=30)
    ax1.set_ylabel('Single Iteration Time', fontsize=35)
    ax1.set_xlabel("PH Horizon", fontsize=35)
    #plt.xlim(20, 100)
    plt.ylim(0, 1.5)
    plt.tick_params(labelsize=30)

    plt.show()


def syn_48_value_reward():
    syn_value_48['Step_asy'] = syn_value_48['Step_asy'].reshape(200, )
    syn_value_48['Step_syn'] = syn_value_48['Step_syn'].reshape(100, )
    syn_value_48['y_low'] = syn_value_48['y_low'].reshape(200, )

    df_for_ref_0 = pd.DataFrame({'Algorithms': 'Ref',
                                   'Position_X': syn_value_48['Step_asy'],
                                   'Position_Y': syn_value_48['y_low'],
                                   'PH': 'Reward Target',
                                   'index': '0'})
    # prediction = 6X8
    df_for_syn_20_0 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Position_X': syn_value_48['Step_syn'],
                                   'Position_Y': syn_value_48['Value_syn_6X8_1'].reshape(100, ),
                                    'PH': 'PH: 6X8',
                                    'index': '0'})
    df_for_syn_20_1 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Position_X': syn_value_48['Step_syn'],
                                   'Position_Y': syn_value_48['Value_syn_6X8_2'].reshape(100, ),
                                    'PH': 'PH: 6X8',
                                    'index': '1'})
    df_for_syn_20_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_6X8_3'].reshape(100, ),
                                 'PH': 'PH: 6X8',
                                    'index': '2'})
    df_for_syn_20_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_6X8_4'].reshape(100, ),
                                 'PH': 'PH: 6X8',
                                    'index': '3'})
    df_for_syn_20_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_6X8_5'].reshape(100, ),
                                    'PH': 'PH: 6X8',
                                    'index': '4'})

    # prediction = 8X6
    df_for_syn_28_0 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Position_X': syn_value_48['Step_syn'],
                                   'Position_Y': syn_value_48['Value_syn_8X6_1'].reshape(100, ),
                                    'PH': 'PH: 8X6',
                                    'index': '0'})
    df_for_syn_28_1 = pd.DataFrame({'Algorithms': 'Syn',
                                   'Position_X': syn_value_48['Step_syn'],
                                   'Position_Y': syn_value_48['Value_syn_8X6_2'].reshape(100, ),
                                    'PH': 'PH: 8X6',
                                    'index': '1'})
    df_for_syn_28_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_8X6_3'].reshape(100, ),
                                 'PH': 'PH: 8X6',
                                    'index': '2'})
    df_for_syn_28_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_8X6_4'].reshape(100, ),
                                 'PH': 'PH: 8X6',
                                    'index': '3'})
    df_for_syn_28_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_8X6_5'].reshape(100, ),
                                    'PH': 'PH: 8X6',
                                    'index': '4'})

    # prediction = 48
    df_for_syn_48_0 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_12X4_1'].reshape(100, ),
                                    'PH': 'PH: 12X4',
                                    'index': '0'})
    df_for_syn_48_1 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_12X4_2'].reshape(100, ),
                                    'PH': 'PH: 12X4',
                                    'index': '1'})
    df_for_syn_48_2 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_12X4_3'].reshape(100, ),
                                    'PH': 'PH: 12X4',
                                    'index': '2'})
    df_for_syn_48_3 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_12X4_4'].reshape(100, ),
                                    'PH': 'PH: 12X4',
                                    'index': '3'})
    df_for_syn_48_4 = pd.DataFrame({'Algorithms': 'Syn',
                                    'Position_X': syn_value_48['Step_syn'],
                                    'Position_Y': syn_value_48['Value_syn_12X4_5'].reshape(100, ),
                                    'PH': 'PH: 12X4',
                                    'index': '4'})


    df_for_20 = df_for_ref_0.append(df_for_syn_20_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_20_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_28_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_48_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_48_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_syn_48_2, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_48_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_syn_48_4, ignore_index=True)


    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_20)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='PH', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=30)
    ax1.set_ylabel('Reward', fontsize=35)
    ax1.set_xlabel("Iteration", fontsize=35)
    plt.xlim(0, 1500)
    plt.ylim(-50, 0)
    plt.tick_params(labelsize=30)
    plt.show()


def asy_48_value_reward():
    asy_value_48['Step_asy'] = asy_value_48['Step_asy'].reshape(200, )
    asy_value_48['y_low'] = asy_value_48['y_low'].reshape(200, )

    df_for_ref_0 = pd.DataFrame({'Algorithms': 'Ref',
                                   'Position_X': asy_value_48['Step_asy'],
                                   'Position_Y': asy_value_48['y_low'],
                                   'PH': 'Reward Target',
                                   'index': '0'})
    # prediction = 6X8
    df_for_asy_20_0 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Position_X': asy_value_48['Step_asy'],
                                   'Position_Y': asy_value_48['Value_asy_6X8_1'].reshape(200, ),
                                    'PH': 'PH: 6X8',
                                    'index': '0'})
    df_for_asy_20_1 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Position_X': asy_value_48['Step_asy'],
                                   'Position_Y': asy_value_48['Value_asy_6X8_2'].reshape(200, ),
                                    'PH': 'PH: 6X8',
                                    'index': '1'})
    df_for_asy_20_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_6X8_3'].reshape(200, ),
                                 'PH': 'PH: 6X8',
                                    'index': '2'})
    df_for_asy_20_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_6X8_4'].reshape(200, ),
                                 'PH': 'PH: 6X8',
                                    'index': '3'})
    df_for_asy_20_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_6X8_5'].reshape(200, ),
                                    'PH': 'PH: 6X8',
                                    'index': '4'})

    # prediction = 8X6
    df_for_asy_28_0 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Position_X': asy_value_48['Step_asy'],
                                   'Position_Y': asy_value_48['Value_asy_8X6_1'].reshape(200, ),
                                    'PH': 'PH: 8X6',
                                    'index': '0'})
    df_for_asy_28_1 = pd.DataFrame({'Algorithms': 'Asy',
                                   'Position_X': asy_value_48['Step_asy'],
                                   'Position_Y': asy_value_48['Value_asy_8X6_2'].reshape(200, ),
                                    'PH': 'PH: 8X6',
                                    'index': '1'})
    df_for_asy_28_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_8X6_3'].reshape(200, ),
                                 'PH': 'PH: 8X6',
                                    'index': '2'})
    df_for_asy_28_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_8X6_4'].reshape(200, ),
                                 'PH': 'PH: 8X6',
                                    'index': '3'})
    df_for_asy_28_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_8X6_5'].reshape(200, ),
                                    'PH': 'PH: 8X6',
                                    'index': '4'})

    # prediction = 48
    df_for_asy_48_0 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_12X4_1'].reshape(200, ),
                                    'PH': 'PH: 12X4',
                                    'index': '0'})
    df_for_asy_48_1 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_12X4_2'].reshape(200, ),
                                    'PH': 'PH: 12X4',
                                    'index': '1'})
    df_for_asy_48_2 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_12X4_3'].reshape(200, ),
                                    'PH': 'PH: 12X4',
                                    'index': '2'})
    df_for_asy_48_3 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_12X4_4'].reshape(200, ),
                                    'PH': 'PH: 12X4',
                                    'index': '3'})
    df_for_asy_48_4 = pd.DataFrame({'Algorithms': 'Asy',
                                    'Position_X': asy_value_48['Step_asy'],
                                    'Position_Y': asy_value_48['Value_asy_12X4_5'].reshape(200, ),
                                    'PH': 'PH: 12X4',
                                    'index': '4'})


    df_for_20 = df_for_ref_0.append(df_for_asy_20_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_20_4, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_3, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_28_4, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_48_0, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_1, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_2, ignore_index=True)
    df_for_20 = df_for_20.append(df_for_asy_48_3, ignore_index=True)
    # df_for_20 = df_for_20.append(df_for_asy_48_4, ignore_index=True)


    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_20)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.lineplot(x="Position_X", y="Position_Y", hue='PH', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2], handles[3]], labels=[labels[0], labels[1], labels[2], labels[3]], loc='lower right', frameon=False, fontsize=22)
    ax1.set_ylabel('Reward', fontsize=35)
    ax1.set_xlabel("Iteration", fontsize=35)
    plt.xlim(0, 2000)
    plt.ylim(-50, 0)
    plt.tick_params(labelsize=30)
    plt.show()


def time_compare():
    MBPG_time_1 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'PH': [20],
                                 'Total Time': [20.8]})
    MBPG_time_2 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'PH': [28],
                                 'Total Time': [30.6]})
    MBPG_time_3 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'PH': [48],
                                 'Total Time': [56.1]})
    MBPG_time_4 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'PH': [60],
                                 'Total Time': [54.4]})
    MBPG_time_5 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'PH': [80],
                                 'Total Time': [106.7]})
    MBPG_time_6 = pd.DataFrame({'Algorithms': 'MBPG',
                                 'PH': [100],
                                 'Total Time': [129.8]})

    Syn_time_1 = pd.DataFrame({'Algorithms': 'Syn',
                                 'PH': [20],
                                 'Total Time': [120]})
    Syn_time_2 = pd.DataFrame({'Algorithms': 'Syn',
                                 'PH': [28],
                                 'Total Time': [105]})
    Syn_time_3 = pd.DataFrame({'Algorithms': 'Syn',
                                 'PH': [48],
                                 'Total Time': [172.5]})
    Syn_time_4 = pd.DataFrame({'Algorithms': 'Syn',
                                 'PH': [60],
                                 'Total Time': [88]})
    Syn_time_5 = pd.DataFrame({'Algorithms': 'Syn',
                                 'PH': [80],
                                 'Total Time': [88]})
    Syn_time_6 = pd.DataFrame({'Algorithms': 'Syn',
                                 'PH': [100],
                                 'Total Time': [75]})

    Asy_time_1 = pd.DataFrame({'Algorithms': 'Asy',
                               'PH': [20],
                               'Total Time': [25.2]})
    Asy_time_2 = pd.DataFrame({'Algorithms': 'Asy',
                               'PH': [28],
                               'Total Time': [34.8]})
    Asy_time_3 = pd.DataFrame({'Algorithms': 'Asy',
                               'PH': [48],
                               'Total Time': [42.7]})
    Asy_time_4 = pd.DataFrame({'Algorithms': 'Asy',
                               'PH': [60],
                               'Total Time': [44.1]})
    Asy_time_5 = pd.DataFrame({'Algorithms': 'Asy',
                               'PH': [80],
                               'Total Time': [33]})
    Asy_time_6 = pd.DataFrame({'Algorithms': 'Asy',
                               'PH': [100],
                               'Total Time': [21.3]})

    df_for_adp = MBPG_time_1.append(MBPG_time_2, ignore_index=True)
    df_for_adp = df_for_adp.append(MBPG_time_3, ignore_index=True)
    df_for_adp = df_for_adp.append(MBPG_time_4, ignore_index=True)
    df_for_adp = df_for_adp.append(MBPG_time_5, ignore_index=True)
    df_for_adp = df_for_adp.append(MBPG_time_6, ignore_index=True)
    df_for_adp = df_for_adp.append(Syn_time_1, ignore_index=True)
    df_for_adp = df_for_adp.append(Syn_time_2, ignore_index=True)
    df_for_adp = df_for_adp.append(Syn_time_3, ignore_index=True)
    df_for_adp = df_for_adp.append(Syn_time_4, ignore_index=True)
    df_for_adp = df_for_adp.append(Syn_time_5, ignore_index=True)
    df_for_adp = df_for_adp.append(Syn_time_6, ignore_index=True)
    df_for_adp = df_for_adp.append(Asy_time_1, ignore_index=True)
    df_for_adp = df_for_adp.append(Asy_time_2, ignore_index=True)
    df_for_adp = df_for_adp.append(Asy_time_3, ignore_index=True)
    df_for_adp = df_for_adp.append(Asy_time_4, ignore_index=True)
    df_for_adp = df_for_adp.append(Asy_time_5, ignore_index=True)
    df_for_adp = df_for_adp.append(Asy_time_6, ignore_index=True)

    df_list_diff_prediction = []

    df_list_diff_prediction.append(df_for_adp)

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8])
    sns.barplot(x="PH", y="Total Time", hue='Algorithms', data=df_list_diff_prediction[0],
                 linewidth=2, palette="bright")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=[handles[0], handles[1], handles[2]], labels=[labels[0], labels[1], labels[2]], loc='upper left', frameon=False, fontsize=30)
    ax1.set_ylabel('Single Iteration Time', fontsize=35)
    ax1.set_xlabel("Prediction Horizon", fontsize=35)
    #plt.xlim(20, 100)
    plt.ylim(0, 200)
    plt.tick_params(labelsize=30)

    plt.show()

if __name__=='__main__':
    m = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/traj_error.mat")
    n = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/value_all.mat")
    adp_value = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/adp_value.mat")
    asy_value = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/asy_value.mat")
    syn_value = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/syn_value.mat")
    asy_value_48 = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/asy_value_48.mat")
    syn_value_48 = loadmat("F:/完成的一些ppt和文档/T3S小论文/dadp/syn_value_48.mat")
    #diff_horizon_sy()
    #diff_horizon_asy()
    #diff_horizon_adp()
    #tracking_performance()
    tracking_error()
    #adp_value_reward()
    #asy_value_reward()
    #syn_value_reward()
    #qushi()
    #syn_48_value_reward()
    # asy_48_value_reward()
    #time_compare()




