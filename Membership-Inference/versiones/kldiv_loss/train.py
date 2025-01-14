import torch
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional as F
import copy
import os
import torchvision.models as models
from torchvision import transforms

# Prepare data for Attack Model
def prepare_attack_data(model,
                        iterator,
                        device,
                        top_k=False,
                        test_dataset=False):
    
    attackX = []
    attackY = []
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in iterator:
            # Move tensors to the configured device
            inputs = inputs.to(device)
            
            # Forward pass through the model
            outputs = model(inputs)
            logits = outputs[0]  # Assuming the first element is logits
            
            # To get class probabilities
            posteriors = F.softmax(logits, dim=1)
            if top_k:
                # Top 3 posterior probabilities(high to low) for train samples
                topk_probs, _ = torch.topk(posteriors, 3, dim=1)
                attackX.append(topk_probs.cpu())
            else:
                attackX.append(posteriors.cpu())

            if test_dataset:
                attackY.append(torch.zeros(posteriors.size(0),dtype=torch.long))
            else:
                attackY.append(torch.ones(posteriors.size(0), dtype=torch.long))
        
    return attackX, attackY

def train_per_epoch(model,
                    train_iterator,
                    criterion,
                    optimizer,
                    device,
                    bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0
    
    model.train()
    for _ , (features, target) in enumerate(train_iterator):
        # Move tensors to the configured device
        features = features.to(device)
        target = target.to(device)
        
        # Forward pass
        outputs = model(features)
        logits = outputs[0]  # Assuming the first element is logits
        log_probs = F.log_softmax(logits, dim=1)
        target_probs = F.one_hot(target, num_classes=log_probs.size(1)).float()
        
        loss = criterion(log_probs, target_probs)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record Loss
        epoch_loss += loss.item()

        # Get predictions for accuracy calculation
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Per epoch validation accuracy calculation
    epoch_acc = correct / total
    epoch_loss = epoch_loss / total

    return epoch_loss, epoch_acc

def val_per_epoch(model,
                val_iterator,
                criterion,
                device,
                bce_loss=False):

    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _,(features,target) in enumerate(val_iterator):
            features = features.to(device)
            target = target.to(device)
            
            outputs = model(features)
            logits = outputs[0]  # Assuming the first element is logits
            log_probs = F.log_softmax(logits, dim=1)
            target_probs = F.one_hot(target, num_classes=log_probs.size(1)).float()
            
            loss = criterion(log_probs, target_probs)
                
            # Record the loss
            epoch_loss += loss.item()
            
            # Check Accuracy
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Per epoch validation accuracy and loss calculation
        epoch_acc = correct / total
        epoch_loss = epoch_loss / total
    
    return epoch_loss, epoch_acc

# Training Attack Model
def train_attack_model(model,
                    dataset,
                    criterion,
                    optimizer,
                    lr_scheduler,
                    device,
                    model_path='./model',
                    epochs=10,
                    b_size=20,
                    num_workers=1,
                    verbose=False,
                    earlystopping=False):
        
    n_validation = 1000 # number of validation samples
    best_valacc = 0
    stop_count = 0
    patience = 5 # Early stopping

    path = os.path.join(model_path,'best_attack_model.ckpt')
        
    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []

    train_X, train_Y = dataset
        
    # Concatenate list of tensors to a single tensor
    t_X = torch.cat(train_X)
    t_Y = torch.cat(train_Y)
  
    # Create Attack Dataset
    attackdataset = TensorDataset(t_X,t_Y)
        
    print('Shape of Attack Feature Data : {}'.format(t_X.shape))
    print('Shape of Attack Target Data : {}'.format(t_Y.shape))
    print('Length of Attack Model train dataset : [{}]'.format(len(attackdataset)))
    print('Epochs [{}] and Batch size [{}] for Attack Model training'.format(epochs,b_size))
        
    # Create Train and Validation Split
    n_train_samples = len(attackdataset) - n_validation
    train_data, val_data = torch.utils.data.random_split(attackdataset, 
                                                               [n_train_samples, n_validation])
        

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=b_size,
                                                shuffle=True,
                                                num_workers=num_workers)
        
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                                  batch_size=b_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)
    
    print('----Attack Model Training------')   
    for i in range(epochs):
            
        train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)
        
        lr_scheduler.step()
        
        print ('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
                 .format(i+1, epochs, train_loss, train_acc*100, valid_loss, valid_acc*100))

        if earlystopping: 
            if best_valacc<=valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                # Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, path)
                stop_count = 0
            else:
                stop_count+=1
                if stop_count >=patience: # early stopping check
                    print('End Training after [{}] Epochs'.format(epochs+1))
                    break
        else: # Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            # Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, path)
            
    return best_valacc

# Training Target and Shadow Model
def train_model(model,
                train_loader,
                val_loader,
                test_loader,
                loss,
                optimizer,
                scheduler,
                device,
                model_path,
                verbose=False,
                num_epochs=50,
                top_k=False,
                earlystopping=False,
                is_target=False):
    
    best_valacc = 0
    patience = 5 # Early stopping
    stop_count= 0
    train_loss_hist = []
    valid_loss_hist = []
    val_acc_hist = []
    
    if is_target:
        print('----Target model training----')
    else:
        print('---Shadow model training----')
    
    # Path for saving best target and shadow models
    target_path = os.path.join(model_path,'best_target_model.ckpt')
    shadow_path = os.path.join(model_path,'best_shadow_model.ckpt')
    
    for epoch in range(num_epochs):
        
        train_loss, train_acc = train_per_epoch(model, train_loader, loss, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, loss, device)

        valid_loss_hist.append(valid_loss)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(valid_acc)

        scheduler.step()

        print ('Epoch [{}/{}], Train Loss: {:.3f} | Train Acc: {:.2f}% | Val Loss: {:.3f} | Val Acc: {:.2f}%'
                   .format(epoch+1, num_epochs, train_loss, train_acc*100, valid_loss, valid_acc*100))
        
        if earlystopping:
            if best_valacc<=valid_acc:
                print('Saving model checkpoint')
                best_valacc = valid_acc
                # Store best model weights
                best_model = copy.deepcopy(model.state_dict())
                if is_target:
                    torch.save(best_model, target_path)
                else:
                    torch.save(best_model, shadow_path)
                stop_count = 0
            else:
                stop_count+=1
                if stop_count >=patience: # early stopping check
                    print('End Training after [{}] Epochs'.format(epoch+1))
                    break
        else: # Continue model training for all epochs
            print('Saving model checkpoint')
            best_valacc = valid_acc
            # Store best model weights
            best_model = copy.deepcopy(model.state_dict())
            if is_target:
                torch.save(best_model, target_path)
            else:
                torch.save(best_model, shadow_path)
    
    if is_target:
        print('----Target model training finished----')
        print('Validation Accuracy for the Target Model is: {:.2f} %'.format(100* best_valacc))
    else:
        print('----Shadow model training finished-----')
        print('Validation Accuracy for the Shadow Model is: {:.2f} %'.format(100* best_valacc))

    if is_target:
        print('----LOADING the best Target model for Test----')
        model.load_state_dict(torch.load(target_path))
    else:
        print('----LOADING the best Shadow model for Test----')
        model.load_state_dict(torch.load(shadow_path))
    
    # As the model is fully trained, time to prepare data for attack model.
    # Training Data for members would come from shadow train dataset, and member inference from target train dataset respectively.
    attack_X, attack_Y = prepare_attack_data(model,train_loader,device,top_k)
    
    # In test phase, we don't need to compute gradients (for memory efficiency)
    print('----Test the Trained Network----')
    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            logits = outputs[0]
            
            # Predictions for accuracy calculations
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Posterior and labels for non-members
            probs_test = F.softmax(logits, dim=1)
            if top_k:
                # Take top K posteriors ranked high ---> low
                topk_t_probs, _ = torch.topk(probs_test, 3, dim=1)
                attack_X.append(topk_t_probs.cpu())
            else:
                attack_X.append(probs_test.cpu())
            attack_Y.append(torch.zeros(probs_test.size(0), dtype=torch.long))

        if is_target:
            print('Test Accuracy of the Target model: {:.2f}%'.format(100 * correct / total))
        else:
            print('Test Accuracy of the Shadow model: {:.2f}%'.format(100 * correct / total)) 
            

    return attack_X, attack_Y
