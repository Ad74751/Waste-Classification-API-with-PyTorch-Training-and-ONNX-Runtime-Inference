import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
from .config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, ONNX_SAVE_PATH, IMG_SIZE, DEVICE
from .dataset import train_dataset, class_weights
from .model import WasteClassifierCNN
from .train_utils import train_epoch, validate


def train_model():
    best_val_acc = 0
    best_model = None

    print(f"Using device: {DEVICE}")
    print("Starting training")

    val_ratio = 0.2
    num_train = int((1 - val_ratio) * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    model = WasteClassifierCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
    if best_model is not None:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'\nBest model saved with validation accuracy: {best_val_acc:.2f}%')
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_SAVE_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f'Model exported to {ONNX_SAVE_PATH}')

if __name__ == "__main__":
    train_model()
