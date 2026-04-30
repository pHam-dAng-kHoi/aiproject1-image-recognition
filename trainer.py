from args import get_args
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# -------------------------------------------------
# SAVE IMAGE + BOUNDING BOXES (works everywhere)
# -------------------------------------------------
def save_image_with_boxes(image, target, save_path):
    img = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in target['boxes']:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title("Training sample")
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
def validate_model(model, val_loader, device):

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    val_epoch_loss = val_loss_sum / val_count
            
    return val_epoch_loss


# -------------------------------------------------
# TRAINING
# -------------------------------------------------
def train_model(model, train_loader, val_loader, device):

    args = get_args()

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    # Create debug image folder
    debug_dir = os.path.join(args.out_dir, "debug_images")
    os.makedirs(debug_dir, exist_ok=True)

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            
            images = [image.to(device=device, dtype=torch.float32) for image in images]

            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            # -------------------------------------------------
            # SAVE EACH IMAGE IN THE BATCH
            # -------------------------------------------------
            for i in range(len(images)):
                save_path = os.path.join(
                    debug_dir,
                    f"epoch_{epoch+1}_batch_{batch_idx+1}_img_{i+1}.png"
                )
                save_image_with_boxes(images[i].cpu(), targets[i], save_path)

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1} / {args.epochs} | "
              f"Train loss: {train_epoch_loss:.4f} | "
              f"Val loss: {val_loss:.4f}")    

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

    # -------------------------------------------------
    # LEARNING CURVE PLOT
    # -------------------------------------------------
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(args.out_dir, 'learning_curve.png')
    plt.savefig(save_path)
    print(f"Learning curve saved to {save_path}")
    plt.close()
