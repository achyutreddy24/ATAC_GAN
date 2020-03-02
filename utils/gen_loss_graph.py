import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../output/MNIST-2071195062666636625/log.csv')

df = df.groupby('Epoch').agg({
        'DLoss': ['mean'],
        'GAdvLoss': ['mean'],
        'GAuxLoss': ['mean'],
        'GTarLoss': ['mean']
})

plt.plot(df.index, df['DLoss'], label='DLoss')
plt.plot(df.index, df['GAdvLoss'], label='G-AdvLoss')
plt.plot(df.index, df['GAuxLoss'], label='G-AuxLoss')
plt.plot(df.index, df['GTarLoss'], label='G-TarLoss')
plt.title('Mean Loss per Epoch')
plt.legend()
plt.savefig('test.jpg')
