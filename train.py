import torch
from torch.utils.data import DataLoader

from dataset.dataloader import (
    DeepfakeDataset,
    create_split
)

from models.full_model import DeepfakeDetector



device="cuda" if torch.cuda.is_available() else "cpu"



# Dataset split

train_samples,val_samples=create_split(
    "data/faces"
)



train_data=DeepfakeDataset(
    "data/faces",
    train_samples
)


val_data=DeepfakeDataset(
    "data/faces",
    val_samples
)




train_loader=DataLoader(

    train_data,

    batch_size=2,

    shuffle=True

)



val_loader=DataLoader(

    val_data,

    batch_size=2

)



# Model

model=DeepfakeDetector()

model=model.to(device)



# Loss

criterion=torch.nn.CrossEntropyLoss()



# Optimizer

optimizer=torch.optim.Adam(

    model.parameters(),

    lr=3e-5

)



epochs=15



for epoch in range(epochs):


    model.train()


    total_loss=0

    correct=0

    total=0



    for videos,labels in train_loader:


        videos=videos.to(device)

        labels=labels.to(device)



        optimizer.zero_grad()



        outputs=model(videos)



        loss=criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



        predictions=torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions==labels
        ).sum().item()


        total += labels.size(0)



    acc=correct/total



    print(
        f"""
Epoch {epoch+1}

Loss:
{total_loss:.4f}

Accuracy:
{acc:.3f}

"""
    )




torch.save(

    model.state_dict(),

    "deepfake_detector.pth"

)