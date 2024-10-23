import time
import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output

def center_crop(tensor, target_size):
    """Center crops a tensor to target_size

    Args:
        tensor (torch.Tensor): Input tensor to crop
        target_size (tuple): Desired output size (H, W)

    Returns:
        torch.Tensor: Center cropped tensor
    """
    h, w = tensor.shape[-2:]
    th, tw = target_size

    i = (h - th) // 2
    j = (w - tw) // 2

    return tensor[..., i:(i + th), j:(j + tw)]

def train(model, opt, loss_fn, epochs, train_loader, test_loader, device):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward pass to get output size
            Y_pred = model(X_batch)

            # Center crop Y_batch to match Y_pred size
            Y_batch_cropped = center_crop(Y_batch, Y_pred.shape[-2:])

            # set parameter gradients to zero
            opt.zero_grad()

            # calculate loss with cropped target
            loss = loss_fn(Y_batch_cropped, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss.item() / len(train_loader)

        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        with torch.no_grad():
            Y_hat = F.sigmoid(model(X_test.to(device))).cpu()

            # Crop test labels to match prediction size
            Y_test_cropped = center_crop(Y_test, Y_hat.shape[-2:])

            clear_output(wait=True)
            for k in range(6):
                plt.subplot(2, 6, k+1)
                plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
                plt.title('Real')
                plt.axis('off')

                plt.subplot(2, 6, k+7)
                plt.imshow(Y_hat[k, 0], cmap='gray')
                plt.title('Output')
                plt.axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
            plt.show()