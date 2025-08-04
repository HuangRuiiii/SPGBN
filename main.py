import argparse
from utils.dataload import load_data
from utils.predict import predict
from model.spgbn import spgbn

parser = argparse.ArgumentParser(description='SPGBN Layerwise Feature Extraction')

parser.add_argument("-t", "--layers_num", type=int, default=3, help="Number of layers in the SPGBN")
parser.add_argument("-k", "--initial_nodes_num", type=int, default=200, help="Number of initial nodes in each layer")
parser.add_argument("-b", "--burnin", type=int, default=100, help="Number of burn-in iterations")
parser.add_argument("-c", "--collection", type=int, default=100, help="Number of collection iterations")
parser.add_argument("-tri", "--train_indices", type=list, default=None, help="Indices of training data")
parser.add_argument("-tei", "--test_indices", type=list, default=None, help="Indices of testing data")
parser.add_argument("--rows", type=int, default=None, help="Number of rows in the data matrix")
parser.add_argument("--cols", type=int, default=None, help="Number of columns in the data matrix")
parser.add_argument("--total_counts", type=int, default=None, help="Total counts in the data matrix")

args = parser.parse_args()

data, label, train_indices, test_indices, _ = load_data("./data/SUBJ_processed_data.pkl")
args.train_indices = train_indices
args.test_indices = test_indices

model = spgbn(data, args)

for t in range(args.layers_num):
    model.train(t)
    theta_aver = model.test()
    predict(label, theta_aver, args)

print("Feature extraction completed successfully.")