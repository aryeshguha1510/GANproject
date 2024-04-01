for epoch in range(num_epochs):
    model.train()
    train_psnr_total = 0.0
    train_ssim_total = 0.0
    
    for images,labels in loaders['train']:
            
            # gives batch data, normalize x when iterate train_loader
        images = images.to(device)  # Assuming device is defined as torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        labels = labels.to(device)
        outputs = model(images)
        loss = nn.MSELoss(outputs, labels)
            
            # clear gradients for this training step
        optimizer.zero_grad()
            
            # backpropagation, compute gradients
        loss.backward()
            
            # apply gradients
        optimizer.step()
            
        for i in range(len(outputs)):
            train_psnr_total += psnr(labels[i].numpy(), outputs[i].detach().numpy())
            train_ssim_total += ssim(labels[i].numpy(), outputs[i].detach().numpy(), multichannel=True)

    train_psnr_avg = train_psnr_total / len(loaders['train'])
    train_ssim_avg = train_ssim_total / len(loaders['train'])
    
   



   
    model.eval()
    val_psnr_total = 0.0
    val_ssim_total = 0.0
    with torch.no_grad():
        for batch in loaders['val']:
            inputs = batch['image']
            targets = batch['target']
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate PSNR and SSIM for validation set
            for i in range(len(outputs)):
                val_psnr_total += psnr(targets[i].numpy(), outputs[i].numpy())
                val_ssim_total += ssim(targets[i].numpy(), outputs[i].numpy(), multichannel=True)

    val_psnr_avg = val_psnr_total / len(loaders['val'])
    val_ssim_avg = val_ssim_total / len(loaders['val'])
    print(f'Epoch [{epoch+1}/{num_epochs}], Train PSNR: {train_psnr_avg:.2f}, Train SSIM: {train_ssim_avg:.4f}, Val PSNR: {val_psnr_avg:.2f}, Val SSIM: {val_ssim_avg:.4f}')
    if (val_psnr_avg > best_val_psnr_avg and val_ssim_avg > best_val_ssim_avg):
        best_val_psnr_avg = val_psnr_avg
        best_val_ssim_avg = val_ssim_avg
        torch.save(model.state_dict(), 'weights.pth')
        print('Weights Saved')