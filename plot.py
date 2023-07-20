import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of file names
file_names = ['0.1_0.07', '0.5_0.01', '0.5_0.03', '0.95_0.01', '0.95_0.001', '0.95_0.07']

# Loop through each file
for file in file_names:
    # Read the reward trace file
    reward_trace = pd.read_csv(f'results/reward/reward_trace_{file}.csv', index_col=0)
    
    # Calculate the rolling mean
    reward_smooth = reward_trace.rolling(window=10, min_periods=0).mean()
    
    # Create the plot
    plt.figure(figsize=(16,9))
    sns.lineplot(data=reward_smooth)
    plt.title('Average performance over the last 125 steps')

    # Save the plot
    plt.savefig(f'results/plots/average_performance_{file}.png')

    # Read the dataframe
    df = pd.read_csv(f'results/pieces/piece_values_{file}.csv')

    # Create a plot of the dataframe excluding the first column
    plt.figure(figsize=(16,9))
    sns.heatmap(df.iloc[:, 1:], annot=True, fmt=".2f")

    # Save the dataframe plot
    plt.savefig(f'results/plots/df_plot_{file}.png')
