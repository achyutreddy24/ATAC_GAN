import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser("Generate Graphs from Model Training Data")
parser.add_argument('--csv_dir', type=str, help='path to the folder that the csv log is in')
args = parser.parse_args()

if (args.csv_dir[len(args.csv_dir)-1] != '/'):
    args.csv_dir += '/'

df = pd.read_csv(args.csv_dir + '/log.csv')

df = df.groupby('Epoch').agg({
        'DLoss': ['mean'],
        'GAdvLoss': ['mean'],
        'GAuxLoss': ['mean'],
        'GTarLoss': ['mean'],
        'GTarLossRaw': ['mean']
})

plt.plot(df.index, df['DLoss'], label='DLoss')
plt.plot(df.index, df['GAdvLoss'], label='G-AdvLoss')
plt.plot(df.index, df['GAuxLoss'], label='G-AuxLoss')
plt.plot(df.index, df['GTarLoss'], label='G-TarLoss')
plt.plot(df.index, df['GTarLossRaw'], label='G-TarLossRaw')
plt.title('Mean Loss per Epoch')
plt.legend()
plt.savefig(args.csv_dir + "/training_graph")
