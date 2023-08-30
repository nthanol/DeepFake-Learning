import time
import torch
import numpy as np
import gc
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from numpy.random import randint

def train_epoch(model, device, trainloader, loss_fn, optimizer, testloader, dataset, epochs=5, default_dtype=torch.FloatTensor, video=False):

    start_time = time.time()
    iters = 0
    running_loss = 0.0

    if video:
        index = randint(len(dataset)) # From the dataset we get a random image
        image, name = getImage(index, dataset) 
        image = image.unsqueeze(0)
        save_image(image, "./Outputs/Video/{}.png".format(name))




    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    for ep in range(epochs):
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for i, (image_batch, _) in enumerate(trainloader): # with "_" we just ignore the labels (the second element of the dataloader tuple)
            if video: # TODO: This only works for the Autoencoder class atm
                model.eval()
                z = model.encode(image)
                output = model.decode(z)
                save_image(output, "./Outputs/Video/{}_{}.png".format(ep, i))
                model.train()


            iters += 1

            # Move tensor to the proper device
            image_batch = image_batch.type(default_dtype).to(device)
            #labels = labels.type(default_dtype).to(device)

            # Encode data
            output = model(image_batch)

            # Evaluate loss
            loss = loss_fn(output, image_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            if i % 1 == 0:
                print('Epoch:{:2d} | Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Average Loss: {:.4f} '.format(ep+1, i, time_lapse, loss.data, running_loss/iters))

            # Print batch loss
            train_loss.append(loss.detach().cpu().numpy())
        gc.collect()
        test_loss = test_epoch(model, device, testloader, loss_fn)
        print('\n EPOCH {}/{} \t Train loss {} \t Test loss {}'.format(ep + 1, epochs, np.mean(train_loss), test_loss))
    return



def test_epoch(model, device, dataloader, loss_fn, default_dtype=torch.FloatTensor):
    # Set evaluation mode for encoder and decoder
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.type(default_dtype).to(device)#image_batch.type(torch.HalfTensor).to(device)
            # Encode data
            output = model(image_batch)

            # Append the network output and the original image to the lists
            conc_out.append(output.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def saveWeights(model, path):
    torch.save(model.state_dict(), path) # model.state_dict(), 'model_weights.pth')

def loadWeights(model, path):
    model.load_state_dict(torch.load(path)) # 'model_weights.pth'))
    model.eval()

def getImage(index, dataset):
    image = dataset[index][0]
    label = dataset[index][1]
    return image, label

def showImage(image):
    """
    Displays image with the option of being saved
    """
    transform = T.ToPILImage()
    img = transform(image)
    img.show()

if __name__ == "__main__":
    print(randint(10))