# Manual Fixes - Step-by-Step Instructions

**What needs to be fixed**: Update the training function to use better blank penalty computation (ratio-based instead of mean-based)

**Why**: The current implementation uses mean-based blank penalty which is less effective. The improved version (already in Cell 20) uses ratio-based penalty which works better.

**Time**: ~5 minutes

---

## Fix 1: Update Training Function (Cell 14)

**Location**: Cell that starts with `def train_epoch(model, train_loader, ...)`

### Step 1.1: Add `compute_blank_ratio` function

**Add this function at the very beginning of Cell 14**, before `def train_epoch`:

```python
def compute_blank_ratio(logits, blank_id=0):
    """Compute ratio of blank predictions (better than mean-based penalty)."""
    predictions = logits.argmax(dim=-1)
    blank_count = (predictions == blank_id).sum().item()
    total_count = predictions.numel()
    return blank_count / total_count if total_count > 0 else 0.0

```

### Step 1.2: Update `train_epoch` function

**Find this section** (around lines 25-30 in Cell 14):

```python
        # Compute CTC loss
        loss = ctc_loss_fn(log_probs, target_flat, lengths, target_lengths)
        
        # Add blank penalty (encourage non-blank predictions)
        if blank_penalty > 0:
            blank_log_probs = log_probs[:, :, 0]  # Blank token probabilities
            blank_penalty_loss = blank_penalty * blank_log_probs.mean()
            loss = loss + blank_penalty_loss
```

**Replace with**:

```python
        # Compute CTC loss
        ctc_loss_value = ctc_loss_fn(log_probs, target_flat, lengths, target_lengths)
        
        # Improved blank penalty: ratio-based (better than mean-based)
        blank_ratio = compute_blank_ratio(logits, blank_id=0)
        blank_penalty_value = blank_penalty * blank_ratio
        loss = ctc_loss_value + blank_penalty_value
```

### Step 1.3: Update variable tracking and return

**Find this section** (around line 35 in Cell 14):

```python
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0
```

**Replace with**:

```python
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased from 1.0
        optimizer.step()
        
        total_loss += loss.item()
        total_ctc_loss += ctc_loss_value.item()
        total_blank_penalty += blank_penalty_value.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_ctc_loss = total_ctc_loss / num_batches if num_batches > 0 else 0.0
    avg_blank_penalty = total_blank_penalty / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_ctc_loss, avg_blank_penalty
```

### Step 1.4: Initialize tracking variables

**Find this section** (around line 5 in Cell 14):

```python
def train_epoch(model, train_loader, optimizer, ctc_loss_fn, device, epoch, blank_penalty=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
```

**Replace with**:

```python
def train_epoch(model, train_loader, optimizer, ctc_loss_fn, device, epoch, blank_penalty=0.5):
    """Train for one epoch with improved blank penalty (ratio-based)."""
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_blank_penalty = 0.0
    num_batches = 0
```

---

## Fix 2: Update Training Loop (Cell 15)

**Location**: Cell that starts with `# Training loop`

### Step 2.1: Unpack tuple from `train_epoch`

**Find this line** (around line 10 in Cell 15):

```python
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, ctc_loss_fn, DEVICE, epoch, BLANK_PENALTY)
    train_losses.append(train_loss)
```

**Replace with**:

```python
    # Train (returns tuple: total_loss, ctc_loss, blank_penalty)
    train_loss, train_ctc_loss, train_blank_penalty = train_epoch(model, train_loader, optimizer, ctc_loss_fn, DEVICE, epoch, BLANK_PENALTY)
    train_losses.append(train_loss)
```

### Step 2.2: Update print statement

**Find this section** (around lines 17-19 in Cell 15):

```python
    # Print metrics
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.2f}%")
```

**Replace with**:

```python
    # Print metrics
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f}, Blank Penalty: {train_blank_penalty:.4f})")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.2f}%")
```

---

## Quick Summary (Copy-Paste Ready)

If you prefer to see the complete changes at once, here's what Cell 14 and Cell 15 should look like:

### Cell 14: Complete `train_epoch` function

```python
def compute_blank_ratio(logits, blank_id=0):
    """Compute ratio of blank predictions (better than mean-based penalty)."""
    predictions = logits.argmax(dim=-1)
    blank_count = (predictions == blank_id).sum().item()
    total_count = predictions.numel()
    return blank_count / total_count if total_count > 0 else 0.0


def train_epoch(model, train_loader, optimizer, ctc_loss_fn, device, epoch, blank_penalty=0.5):
    """Train for one epoch with improved blank penalty (ratio-based)."""
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_blank_penalty = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        features = batch['features'].to(device)
        targets = batch['targets']
        lengths = batch['lengths']
        gloss_ids = batch['gloss_ids'].to(device)
        
        # Forward pass
        logits = model(features, lengths)
        
        # Prepare CTC inputs
        # logits: (batch_size, seq_len, vocab_size) -> (seq_len, batch_size, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (seq_len, batch_size, vocab_size)
        
        # Targets for CTC
        target_lengths = torch.LongTensor([len(t) for t in targets]).to(device)
        target_flat = torch.cat(targets).to(device)
        
        # Compute CTC loss
        ctc_loss_value = ctc_loss_fn(log_probs, target_flat, lengths, target_lengths)
        
        # Improved blank penalty: ratio-based (better than mean-based)
        blank_ratio = compute_blank_ratio(logits, blank_id=0)
        blank_penalty_value = blank_penalty * blank_ratio
        loss = ctc_loss_value + blank_penalty_value
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased from 1.0
        optimizer.step()
        
        total_loss += loss.item()
        total_ctc_loss += ctc_loss_value.item()
        total_blank_penalty += blank_penalty_value.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_ctc_loss = total_ctc_loss / num_batches if num_batches > 0 else 0.0
    avg_blank_penalty = total_blank_penalty / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_ctc_loss, avg_blank_penalty


# ... (keep the validate function as-is) ...
```

### Cell 15: Updated training loop

```python
for epoch in range(1, NUM_EPOCHS + 1):
    # Train (returns tuple: total_loss, ctc_loss, blank_penalty)
    train_loss, train_ctc_loss, train_blank_penalty = train_epoch(model, train_loader, optimizer, ctc_loss_fn, DEVICE, epoch, BLANK_PENALTY)
    train_losses.append(train_loss)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, ctc_loss_fn, DEVICE, epoch)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Print metrics
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f}, Blank Penalty: {train_blank_penalty:.4f})")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.2f}%")
    
    # ... (rest of the training loop stays the same) ...
```

---

## What These Fixes Do

1. **Ratio-based blank penalty**: Instead of penalizing mean blank log-probability, we penalize the actual ratio of blank predictions. This is more effective.

2. **Better monitoring**: Returns separate CTC loss and blank penalty values so you can see what's contributing to the total loss.

3. **Increased gradient clipping**: Changed from 1.0 to 5.0 for more stable training with larger models.

---

## After Making Fixes

1. ✅ Save the notebook
2. ✅ Verify cells run without errors (use "Run" to test syntax)
3. ✅ Upload to Colab and run

The training should now work better, especially with Stage 1 pre-trained weights loaded!
