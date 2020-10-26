%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

def train_network(epochs = 3,print_every = 10):
    steps = 0
    running_loss = 0

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/print_every) 
                test_losses.append(test_loss/len(validloaders) )
                               
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Valid loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {test_loss/len(validloaders):.3f}.. "
                  f"Valid accuracy: {(accuracy/len(validloaders))*100:.3f}%")
            running_loss = 0
            model.train()
                                 
