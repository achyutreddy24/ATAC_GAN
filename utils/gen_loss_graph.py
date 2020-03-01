import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../output/MNIST-6886085045213099351/log.csv')

df = df.groupby('Epoch').agg({
        'DLoss': ['mean'],
        'AdvLoss': ['mean'],
        'AuxLoss': ['mean'],
        'TarLoss': ['mean']
})

plt.plot(df.index, df['DLoss'], label='DLoss')
plt.plot(df.index, df['AdvLoss'], label='G-AdvLoss')
plt.plot(df.index, df['AuxLoss'], label='G-AuxLoss')
plt.plot(df.index, df['TarLoss'], label='G-TarLoss')
plt.title('Mean Loss per Epoch')
plt.legend()
plt.savefig('test.jpg')
