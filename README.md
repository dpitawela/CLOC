# CLOC: Contrastive Learning for Ordinal Classification with Multi-Margin N-pair Loss

## Overview

CLOC is a contrastive loss function designed for ordinal classification tasks. It supports both multiple and single margin setups and can be integrated easily into any PyTorch training loop.

[**Paper**](#)

---

## Initialization

To use CLOC, you need to specify the following:

- `n_classes`: Number of classes in the dataset.
- `summaryWriter`: (Optional) Pass a SummaryWriter if you want to log the margins during training.
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

#### 1. Random Initialization

To initialize margins with random values between 0.5 and 1.0:

```python
learnable_map = None
```

#### 2. Specific Learnable Values

To initialize with specific values and allow them to be learned:

```python
# Use values of your choice
learnable_map = [
    ['learnable', 0.12],
    ['learnable', 0.25],
    ['learnable', 0.51],
    ['learnable', 0.32]
]
```

#### 3. Mix of Fixed and Learnable Margins

To fix some margins and learn others:

```python
learnable_map = [
    ['fixed', 0.12],
    ['learnable', 0.25],
    ['fixed', 0.51],
    ['learnable', 0.32]
]
```

---

### For Single Margin

#### 1. Random Initialization

To initialize with a random value between 0.5 and 1.0:

```python
param_map = None
```

#### 2. Learnable Single Margin

To initialize with a specific learnable value:

```python
param_map = [
    ['learnable', 0.47]
]
```

#### 3. Fixed Single Margin

To fix the margin value and prevent it from being updated:

```python
param_map = [
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
    learnable_map=param_map
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