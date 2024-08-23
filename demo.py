import os
import torch
import numpy as np
import time
import random
from sklearn.model_selection import KFold
import argparse
import timeit
from dataset import prepare_data
import psutil
import torch
from models.gconvGRU import GConvGRUModel
from memory_capacity_utils import gen_lag_data, compute_memory_capacity_vectorized, get_mem_cap_from_model

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
else:
    device = torch.device("cpu")
    print('running on CPU')
    
def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-num_folds', type=int, default=5, help='cv number')
    parser.add_argument('--num_timepoints', type=int, default=3,
                        help='Number of timepoints')
    parser.add_argument('-num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--memcap_coef', type=float, default=0.0001, help="Memory Capacity Loss Coefficient")
    parser.add_argument('-max_lag', type=int, default=35, help='Lag tao for memory capacity signals')
    parser.add_argument('-save_path',type=str,default = '/vol/bitbucket/sx420/4D-FedGNN Plus/results/reservoir_gconvGRU_oasis/',help='Path to the saved results')
    args, _ = parser.parse_known_args()
    return args


def create_directory_if_not_exists(directory):
    """
    Checks if a specified directory exists, and creates it if it doesn't.

    Args:
    - directory (str): Path of the directory to check and potentially create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' was created.")
    else:
        print(f"Directory '{directory}' already exists.")
        

def validation(args, model, validation_subjects, mem_cap_data, X_train, y_train, X_test, y_test):
    mael = torch.nn.L1Loss().to(device)
    tp = torch.nn.MSELoss().to(device)
    
    val_mae_loss = np.zeros(args.num_timepoints - 1)
    val_tp_loss = np.zeros(args.num_timepoints - 1)
    mem_cap = np.zeros(args.num_timepoints - 1)
    predicted = np.zeros((validation_subjects.shape[0], args.num_timepoints - 1, 35, 35))
    actual = np.zeros((validation_subjects.shape[0], args.num_timepoints - 1, 35, 35))
    
    model.eval()

    with torch.no_grad():
        for n_subject, data in enumerate(validation_subjects):
            input = data[0]
            for t in range(args.num_timepoints - 1):
                pred = model(input)
                val_mae_loss[t] += mael(pred, data[t + 1])
                val_tp_loss[t] += tp(pred.sum(dim=-1), data[t + 1].sum(dim=-1))
                input = pred
                
                pred_mem_cap = get_mem_cap_from_model(model, pred, 
                                                      X_train, y_train, X_test, y_test)
                actual_mem_cap = torch.tensor(mem_cap_data[n_subject, t + 1]).to(device)
                mem_cap[t] += torch.abs(pred_mem_cap - actual_mem_cap)

                predicted[n_subject, t] = pred.cpu().detach().numpy()
                actual[n_subject, t] = data[t + 1].cpu().detach().numpy()
                

    avg_val_mae_loss = val_mae_loss/len(validation_subjects)
    avg_val_tp_loss = val_tp_loss/len(validation_subjects)
    avg_val_mae_mem_cap = mem_cap/len(validation_subjects)

    return avg_val_mae_loss, avg_val_tp_loss, avg_val_mae_mem_cap, predicted, actual

def train(args, dataset, actual_mem_caps, X_train_res, y_train_res, X_test_res, y_test_res):
    torch.manual_seed(1)
    input_weights = (torch.rand((35, 1), dtype=torch.float64) * 2.0 - 1.0).to(device)

    indexes = range(args.num_folds)
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=manual_seed)
    dataset = dataset.to(device)
    f = 0

    for train, test in kfold.split(range(dataset.shape[0])):
        print(
                f'------------------------------------Fold [{f + 1}/{args.num_folds}]-----------------------------------------')

        train_data = dataset[train]
        test_data = dataset[test]
        train_mem_cap = actual_mem_caps[train]
        test_mem_cap = actual_mem_caps[test]

        validation_split = int(0.8 * len(train_data))
        train_subjects = train_data[:validation_split]
        train_mem_cap_subjects = train_mem_cap[:validation_split]

        validation_subjects = train_data[validation_split:]
        validation_mem_cap_subjects = train_mem_cap[:validation_split]

        model = GConvGRUModel(device=device, input_weights=input_weights, input_scaling=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        mael = torch.nn.L1Loss().to(device)
        tp = torch.nn.MSELoss().to(device)

        # Start measuring the epochs time
        epochs_start = time.time()
        for epoch in range(args.num_epochs):

            print(f'Epoch [{epoch + 1}/{args.num_epochs}]')
            # Set the model in training mode
            model.train()

            # this is our loss for all the data
            mae_loss_overall = []
            tp_loss_overall = []
            mae_mem_cap_overall = []

            # loop through the data batches
            for data_id, data in enumerate(train_subjects):

                # zero the gradients
                optimizer.zero_grad()
                mae_loss = 0
                tp_loss = 0
                mem_cap_loss = 0

                # loop through the time dependent adj matrices in the batches
                for t in range(args.num_timepoints - 1):
                    pred, output_sig = model(data[t], X_train_res, y_train_res, X_test_res)

                    real = data[t + 1]

                    mae_loss += mael(pred, real)

                    # Topological Loss
                    tp_loss += tp(pred.sum(dim=-1), real.sum(dim=-1))

                    # MAE between predicted graph's mem cap and actual graph's mem cap
                    predicted_mem_cap = compute_memory_capacity_vectorized(output_sig, y_test_res)
                    actual_mem_cap = torch.tensor(train_mem_cap_subjects[data_id, t + 1], requires_grad=True).to(device)
                    mem_cap_loss += mael(predicted_mem_cap, actual_mem_cap)


                # Calculate the total MAE Loss for the current batch
                mae_loss = mae_loss / (args.num_timepoints - 1)

                # Calculate the total TP Loss for the current batch
                tp_loss = tp_loss / (args.num_timepoints - 1)

                # Calculate the total MAE between Mem Cap Loss for the current batch
                mem_cap_loss = mem_cap_loss / (args.num_timepoints - 1)

                # Append to the total MAE Loss
                mae_loss_overall.append(mae_loss.item())
                tp_loss_overall.append(tp_loss.item())
                mae_mem_cap_overall.append(mem_cap_loss.item())

                total_loss = mae_loss + args.memcap_coef * mem_cap_loss 
                # Update the weights of the neural network
                total_loss.backward()
                optimizer.step()

            mae_loss_overall = np.mean(np.array(mae_loss_overall))
            tp_loss_overall = np.mean(np.array(tp_loss_overall))
            mae_mem_cap_overall = np.mean(np.array(mae_mem_cap_overall))
            print(f"[Train] MAE Loss: {mae_loss_overall}, TP Loss: {tp_loss_overall}, MAE of Mem Caps Loss: {mae_mem_cap_overall}")

            avg_val_mae_loss, avg_val_tp_loss, avg_val_mae_mem_cap, _, _ = validation(args, model, validation_subjects, validation_mem_cap_subjects, X_train_res, y_train_res, X_test_res, y_test_res)
            print(f"[Validate] MAE Loss Across Timepoints: {avg_val_mae_loss}")
            print(f"[Validate] TP Loss Across Timepoints: {avg_val_tp_loss}")
            print(f"[Validate] MAE of Mem Caps Across Timepoints: {avg_val_mae_mem_cap}")


        epochs_end = time.time() - epochs_start
        print()
        print(f'epochs finished with time:{epochs_end}')
        print()
        process = psutil.Process(os.getpid())
        print(f"Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        print()

        avg_test_mae_loss, avg_test_tp_loss, avg_test_mem_cap, predicted, original = validation(args, model, test_data, test_mem_cap,
                                                                                  X_train_res, y_train_res, X_test_res, y_test_res)
        print(f"[Test] MAE Loss Across Timepoints: {avg_test_mae_loss}")
        print(f"[Test] TP Loss Across Timepoints: {avg_test_tp_loss}")
        print(f"[Test] MAE of Mem Caps Across Timepoints: {avg_test_mem_cap}")
        np.save(args.save_path+f"test_mae_losses/mae_test_loss_fold_{f}", avg_test_mae_loss)
        np.save(args.save_path+f"test_tp_losses/tp_test_loss_fold_{f}", avg_test_mae_loss)
        np.save(args.save_path+f"test_memcap_losses/memcap_test_loss_fold_{f}", avg_test_mem_cap)
        np.save(args.save_path+f"test_predicted/predicted_fold_{f}", predicted)
        np.save(args.save_path+f"test_original/original_fold_{f}", original)

        torch.save(model.state_dict(),
                   args.save_path +f'trained_models/model_fold_{f}')
        f += 1


if __name__ == "__main__":
    args = get_args()
    dataset = np.load("datasets/multivariate_simulation_data_2.npy")
    dataset = torch.from_numpy(dataset).squeeze()
    dataset = dataset.type(torch.FloatTensor)
   
    manual_seed = 777
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Reservoir signals
    X_train_res_np, y_train_res_np = gen_lag_data(1000, 41, args.max_lag)
    X_test_res_np, y_test_res_np = gen_lag_data(500, 42, args.max_lag)
    X_train_res = torch.from_numpy(X_train_res_np).unsqueeze(1).to(device, dtype=torch.float64)
    X_test_res = torch.from_numpy(X_test_res_np).unsqueeze(1).to(device, dtype=torch.float64)
    y_train_res = torch.from_numpy(y_train_res_np).to(device, dtype=torch.float64)
    y_test_res = torch.from_numpy(y_test_res_np).to(device, dtype=torch.float64)
    
    # Preprocessing
    print("Calculating memory capacities...")
    torch.manual_seed(1)
    input_weights = (torch.rand((35, 1), dtype=torch.float64) * 2.0 - 1.0).to(device)
    model = GConvGRUModel(device=device, input_weights=input_weights, input_scaling=1e-6).to(device)
    actual_mem_caps = np.zeros((dataset.shape[0], dataset.shape[1]))
    for n_subjects in range(dataset.shape[0]):
        for n_t in range(dataset.shape[1]):
            actual_mem_caps[n_subjects, n_t] = get_mem_cap_from_model(model, dataset[n_subjects, n_t, :, :].to(device), X_train_res, y_train_res, X_test_res, y_test_res).item()
    
    # Training
    train(args, dataset, actual_mem_caps, X_train_res, y_train_res, X_test_res, y_test_res)
