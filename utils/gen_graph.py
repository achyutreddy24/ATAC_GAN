import argparse
import matplotlib.pyplot as plt
import pandas as pd

# Hack to make all links absolute
from sys import path
import os
p = os.path.abspath("")
root_dir = "ATAC_GAN" # Root directory of project
p = p[:p.rindex(root_dir)+len(root_dir)]
if p not in path:
    path.append(p)

parser = argparse.ArgumentParser("Generate Graphs from Model Training Data")
parser.add_argument('--csv_dir', default = "", type=str, help='path to the folder that the csv log is in')
parser.add_argument('--name', default = "", type=str, help='Checks output folder for name')
parser.add_argument("--g", default=False, action="store_true", help="Plot all generator loss info")
parser.add_argument("--d", default=False, action="store_true", help="Plot all discriminator loss info")
parser.add_argument("--acc", default=False, action="store_true", help="Plot all accuracy info")
parser.add_argument("--all", default=False, action="store_true", help="Plot all info")
parser.add_argument("--loss", default=False, action="store_true", help="Plot all loss info")
parser.add_argument("--valid", default=False, action="store_true", help="Plot all validity info")
args = parser.parse_args()

# Input validation
if args.csv_dir == "" and args.name == "":
    raise AssertionError # At least one of these values (csv_dir, name) should be provided
elif args.csv_dir != "" and args.name != "":
    temp_dir = p+"/output/" + args.name
    args.csv_dir = os.path.abspath(args.csv_dir)
    if args.csv_dir != temp_dir:
        raise AssertionError # Name and directory both provided but don't match
elif args.name != "":
    args.csv_dir=p+"/output/" + args.name
else:
    args.run_name = args.csv_dir.split('/')[-1]
    args.csv_dir = os.path.abspath(args.csv_dir)

if (not os.path.isdir(args.csv_dir+"/graphs")):
    os.mkdir(args.csv_dir+"/graphs")

output_dir = args.csv_dir+"/graphs"

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
plt.savefig(output_dir + "/graph_" + ("_g" if args.g else "") + ("_d" if args.d else "") + ("_acc" if args.acc else "") + ("_all" if args.all else "") + ("_loss" if args.loss else ""))
