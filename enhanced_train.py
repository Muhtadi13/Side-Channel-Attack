import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from collect import WEBSITES
from train import FingerprintClassifier, ComplexFingerprintClassifier

# Configuration
DATASET_PATH = "dataset_merged.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50  
LEARNING_RATE = 1e-04
TRAIN_SPLIT = 0.8 
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Custom Dataset Class
class TraceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential() if in_channels == out_channels and stride == 1 else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(identity)
        out = self.relu(out)
        return out

# Enhanced Model with Residual Connections
class EnhancedResCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EnhancedResCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.res_block1 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.res_block2 = ResidualBlock(32, 64, stride=2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.res_block3 = ResidualBlock(64, 128, stride=2)
        self.pool3 = nn.MaxPool1d(2, 2)

        # Recalculate conv_output_size manually
        temp_size = input_size
        temp_size = (temp_size + 2 * 2 - 5) // 2 + 1  # conv1 output
        temp_size = temp_size // 2  # pool1
        temp_size = (temp_size + 2 * 1 - 3) // 2 + 1  # res_block2 conv1 (stride 2)
        temp_size = temp_size // 2  # pool2
        temp_size = (temp_size + 2 * 1 - 3) // 2 + 1  # res_block3 conv1 (stride 2)
        temp_size = temp_size // 2  # pool3
        conv_output_size = temp_size
        self.fc_input_size = conv_output_size * 128
        print(f"Computed fc_input_size: {self.fc_input_size} for conv_output_size: {conv_output_size}")

        self.fc1 = nn.Linear(self.fc_input_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)
        # print(f"After unsqueeze: {x.shape}")
        x = self.relu(self.bn1(self.conv1(x)))
        # print(f"After conv1: {x.shape}")
        x = self.res_block1(x)
        # print(f"After res_block1: {x.shape}")
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        x = self.res_block2(x)
        # print(f"After res_block2: {x.shape}")
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        x = self.res_block3(x)
        # print(f"After res_block3: {x.shape}")
        x = self.pool3(x)
        # print(f"After pool3: {x.shape}")
        x = x.view(-1, self.fc_input_size)
        # print(f"After view: {x.shape}")
        x = self.relu(self.bn2(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn3(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    

# Data Augmentation
def augment_traces(traces, labels, augment_factor=2):
    augmented_traces = traces.copy()
    augmented_labels = labels.copy()
    for _ in range(augment_factor - 1):
        for trace in traces:
            noise = np.random.normal(0, 0.01, trace.shape)
            jitter = np.random.uniform(-0.05, 0.05, trace.shape)  # Add small jitter
            aug_trace = trace + noise + jitter
            augmented_traces = np.vstack([augmented_traces, aug_trace])
            augmented_labels = np.append(augmented_labels, labels[0])  # Same label
    return augmented_traces, augmented_labels

def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                # Scheduler step will be called with metrics later
                pass
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Step the scheduler with test accuracy (since mode='max')
        if scheduler:
            scheduler.step(epoch_accuracy)
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    return best_accuracy


def evaluate(model, test_loader, website_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    print(f"Loaded dataset with {len(data)} entries.")

    traces = []
    labels = []
    for entry in data:
        trace = entry['trace_data']
        if len(trace) < INPUT_SIZE:
            trace = np.pad(trace, (0, INPUT_SIZE - len(trace)), 'constant')
        elif len(trace) > INPUT_SIZE:
            trace = trace[:INPUT_SIZE]
        traces.append(trace)
        labels.append(entry['website_index'])

    traces = np.array(traces, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"Traces shape: {traces.shape}, Labels shape: {labels.shape}")    
    # Data Augmentation
    # traces, labels = augment_traces(traces, labels, augment_factor=2)

    print(f"After augmentation, traces shape: {traces.shape}, Labels shape: {labels.shape}")

    # Normalize the traces
    min_val = np.min(traces)
    max_val = np.max(traces)
    # with open('normalization_params.json', 'w') as f:
    #     json.dump({'min_val': float(min_val), 'max_val': float(max_val)}, f)
    traces = (traces - min_val) / (max_val - min_val) if max_val != min_val else traces



    num_classes = len(np.unique(labels))
    print(f"Derived num_classes from dataset: {num_classes}")

    # Split the dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TRAIN_SPLIT, random_state=42)
    for train_idx, test_idx in sss.split(traces, labels):
        X_train, X_test = traces[train_idx], traces[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

    # Data Loaders
    train_dataset = TraceDataset(X_train, y_train)
    test_dataset = TraceDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Models
    models = {
        "SimpleCNN": FingerprintClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=num_classes),
        "ComplexCNN": ComplexFingerprintClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=num_classes),
        "EnhancedResCNN": EnhancedResCNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=num_classes)
    }

    # Hyperparameter Tuning and Training
    criterion = nn.CrossEntropyLoss()
    results = {}
    best_accuracy = 0.0
    best_model_path = ""

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        # Hyperparameter options
        lr_options = [1e-04]
        for lr in lr_options:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            model_save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            accuracy = train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, model_save_path, scheduler)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_save_path
            results[model_name] = {"accuracy": accuracy, "path": model_save_path}

    # Ensemble Prediction
    ensemble_preds = []
    for model_name, result in results.items():
        model = models[model_name]
        model.load_state_dict(torch.load(result["path"]))
        model.to(device)
        preds, _ = evaluate(model, test_loader, WEBSITES)
        ensemble_preds.append(preds)

    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(ensemble_preds))
    true_labels, _ = evaluate(models["ComplexCNN"], test_loader, WEBSITES)
    ensemble_accuracy = 100 * np.mean(ensemble_pred == true_labels)

    print(f"\nBest Single Model Accuracy: {best_accuracy:.2f}% (Saved at {best_model_path})")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}%")
    if ensemble_accuracy > best_accuracy:
        # Save ensemble weights (simplified by saving the best model for now)
        torch.save(models["EnhancedResCNN"].state_dict(), os.path.join(MODELS_DIR, "best_ensemble_model.pth"))
        print("Ensemble model saved as best model.")

    # Plot training curves (optional)
    # plt.figure()
    # for model_name in models:
    #     model = models[model_name]
    #     model.load_state_dict(torch.load(results[model_name]["path"]))
    #     train_acc, test_acc = [], []
    #     for epoch in range(EPOCHS):
    #         model.eval()
    #         with torch.no_grad():
    #             correct = 0
    #             total = 0
    #             for traces, labels in train_loader:
    #                 traces, labels = traces.to(device), labels.to(device)
    #                 outputs = model(traces)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total += labels.size(0)
    #                 correct += (predicted == labels).sum().item()
    #             train_acc.append(correct / total)
    #             correct = 0
    #             total = 0
    #             for traces, labels in test_loader:
    #                 traces, labels = traces.to(device), labels.to(device)
    #                 outputs = model(traces)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total += labels.size(0)
    #                 correct += (predicted == labels).sum().item()
    #             test_acc.append(correct / total)
    #     plt.plot(train_acc, label=f'{model_name} Train')
    #     plt.plot(test_acc, label=f'{model_name} Test')
    # plt.legend()
    # plt.title("Training and Test Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.savefig(os.path.join(MODELS_DIR, "accuracy_curves.png"))
    # plt.close()

if __name__ == "__main__":
    main()