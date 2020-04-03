import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser("Generate Graphs from Model Training Data")
parser.add_argument('--csv_dir', type=str, help='path to the folder that the csv log is in')
parser.add_argument("--g", default=False, action="store_true", help="Plot all generator loss info")
parser.add_argument("--d", default=False, action="store_true", help="Plot all discriminator loss info")
parser.add_argument("--acc", default=False, action="store_true", help="Plot all accuracy info")
parser.add_argument("--all", default=False, action="store_true", help="Plot all info")
parser.add_argument("--loss", default=False, action="store_true", help="Plot all loss info")
parser.add_argument("--valid", default=False, action="store_true", help="Plot all validity info")
args = parser.parse_args()

if (args.csv_dir[len(args.csv_dir)-1] != '/'):
    args.csv_dir += '/'

df = pd.read_csv(args.csv_dir + '/log.csv')
cols = list(df.columns)
cols.remove("Epoch")
cols.remove("Batch")

df = df.groupby('Epoch').mean()

for label in cols:
    acc = "Acc" in label
    loss = "Loss" in label
    valid = "Valid" in label
    if (
        (args.all) or
        (label[0] == 'G' and args.g and not acc) or 
        (label[0] == 'D' and args.d and not acc) or
        (acc and args.acc) or
        (loss and args.loss) or
        (valid and args.valid)
    ):
        plt.plot(df.index, df[label], label=label)
    
plt.xlabel('Epoch')
plt.title('Mean per Epoch')
plt.legend()
plt.axes().yaxis.grid(True)
plt.savefig(args.csv_dir + "/training_graph" + ("_g" if args.g else "") + ("_d" if args.d else "") + ("_acc" if args.acc else "") + ("_all" if args.all else "") + ("_loss" if args.loss else ""))
