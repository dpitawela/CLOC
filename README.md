# CLOC: Contrastive Learning for Ordinal Classification with Multi-Margin N-pair Loss

## Overview

CLOC is a contrastive loss function designed for ordinal classification tasks. It supports both multiple and single margin setups and can be integrated easily into PyTorch training loop.

[**Paper**](#)

---

## Initialization

To use CLOC, you need to specify the following:

- `n_classes`: Number of classes in the dataset.
- `summaryWriter`: (Optional) Pass a SummaryWriter object from `torch.utils.tensorboard`, if you want to log the margins during training. If you choose to use it, make sure to pass `global_step` inside the training loop.
- `learnable_map`: Pass a margin setting (explained below).

### Multiple Margins

Use when you want different margins between each class.

```python
margin_criterion = OrdinalContrastiveLoss_mm(
    n_classes=5, 
    device=device, 
    summaryWriter=writer, 
    learnable_map=None
)
```

### Single Margin

Use when you want a single margin value shared across all class pairs.

```python
margin_criterion = OrdinalContrastiveLoss_sm(
    n_classes=5, 
    device=device, 
    summaryWriter=writer, 
    learnable_map=None
)
```

---

## Margin Parameter Setup

### For Different Margins (Multiple)

Use when you want different margins between class pairs.

#### 1. Random Initialization

To initialize all margins with random values between 0.5 and 1.0:

```python
learnable_map = None
```

#### 2. Initialize With Specific Values

For a dataset with five classes, to initialize some margins with specific values:

```python
# Use values of your choice
learnable_map = [
    ['learnable', None],
    ['learnable', 0.25],
    ['learnable', 0.51],
    ['learnable', None]
]
```
`None` values are randomly initialised between 0.5 and 1.0. All margins are being updated during training as they are `learnable`.


#### 3. Mix of Fixed and Learnable Margins

To fix some margins to prevent them from being updated during training while learning others:

```python
learnable_map = [
    ['fixed', 0.12],
    ['learnable', None],
    ['fixed', 0.40],
    ['learnable', 0.32]
]
```
`fixed` margins do not change during training.

---

### For Single Margin

Use when you want a single margin value shared across all class pairs.

#### 1. Random Initialization

To initialize with a random value between 0.5 and 1.0:

```python
learnable_map = None
```

#### 2. Learnable Single Margin

To initialize with a specific learnable value:

```python
learnable_map = [
    ['learnable', 0.47]
]
```

#### 3. Fixed Single Margin

To fix the margin value and prevent it from being updated:

```python
learnable_map = [
    ['fixed', 0.47]
]
```

---

## Usage

CLOC can be used like any standard loss function in your training loop.

```python
# Initialize the margin-based contrastive loss
margin_criterion = OrdinalContrastiveLoss_mm(
    n_classes=5,
    device=device,
    summaryWriter=writer,
    learnable_map=learnable_map
)

# Classification loss
cls_criterion = nn.CrossEntropyLoss()

# Optimizer includes both model and margin parameters
optimizer = optim.Adam(
    list(model.parameters()) + list(margin_criterion.parameters())
)

# Training loop
for i, (imgs, labels) in enumerate(trainloader):
    model.train()
    imgs = imgs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    output = model(imgs)

    loss = cls_criterion(output, labels) + margin_criterion(output, labels, global_step)
    loss.backward()
    optimizer.step()
    global_step += 1
```

### Training Phases

CLOC employs a two-phase training strategy:

- **Phase One:** In this phase, both the model parameters and the parameters of the `OrdinalContrastiveLoss` (i.e. the margins) are optimized jointly (as in the above training loop).
- **Phase Two:** In this phase, only the model parameters are optimized, while the margin values learned in the Phase One are kept fixed. The margin values from Phase One are reused by passing them through a `learnable_map`, ensuring they remain fixed during training.

Refer the paper for more details.