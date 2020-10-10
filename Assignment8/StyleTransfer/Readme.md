# Style Transfer with Deep Neural Networks

In this notebook, we have recreated a style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch.

We have also taken inspiration from the implementation of this paper from the following blog: [Style Transfer using Pytorch](https://medium.com/analytics-vidhya/style-transfer-pytorch-84cf2e9ba86d) written by Alejandro Diaz

In this paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers. In the image below, the convolutional layers are named by stack and their order in the stack. Conv_1_1 is the first convolutional layer that an image is passed through, in the first stack. Conv_2_1 is the first convolutional layer in the second stack. The deepest convolutional layer in the network is conv_5_4.

![Image](https://github.com/gdeotale/E4P2/blob/master/Assignment8/StyleTransfer/notebook_ims/vgg19_convlayers.png)

### Separating Style and Content

Style transfer relies on separating the content and style of an image. Given one content image and one style image, we aim to create a new, target image which should contain our desired content and style components:

objects and their arrangement are similar to that of the content image
style, colors, and textures are similar to that of the style image
An example is shown below:

![Image](https://github.com/gdeotale/E4P2/blob/master/Assignment8/StyleTransfer/images/results/Dancing-Stairs.jpg)

### Loss and Weights

#### Individual Layer Style Weights
We have the option to weight the style representation at each relevant layer. We have used a range between 0-1 to weight these layers. By weighting earlier layers (conv1_1 and conv2_1) more, you can expect to get larger style artifacts in your resulting, target image. Choosing to weight later layers, we get more emphasis on smaller features. This is because each layer is a different size and together they create a multi-scale style representation!

### Content and Style Weight
Just like in the paper, we define an alpha (content_weight) and a beta (style_weight). This ratio will affect how stylized your final image is. It's recommended that you leave the content_weight = 1 and set the style_weight to achieve the ratio you want.

### Gram Matrix
The output of every convolutional layer is a Tensor with dimensions associated with the batch_size, a depth, d and some height and width (h, w). The Gram matrix of a convolutional layer can be calculated as follows:

Get the depth, height, and width of a tensor using batch_size, d, h, w = tensor.size
Reshape that tensor so that the spatial dimensions are flattened
Calculate the gram matrix by multiplying the reshaped tensor by it's transpose
Note: You can multiply two matrices using torch.mm(matrix1, matrix2).

### Loss and Weights
#### Individual Layer Style Weights
We have weighed the style representation at each relevant layer. We have used range between 0-1 to weight these layers. By weighting earlier layers (conv1_1 and conv2_1) more, we expect to get larger style artifacts in target image. By weighing later layers, we get more emphasis on smaller features. This is because each layer is a different size and together they create a multi-scale style representation!

#### Content and Style Weight
Just like in the paper, we define an alpha (content_weight) and a beta (style_weight). This ratio will affect how stylized your final image is. It's recommended that you leave the content_weight = 1 and set the style_weight to achieve the ratio you want.

## Updating the Target & Calculating Losses
We have used 5000 steps for good results.

Inside the iteration loop, we have calculated the content and style losses and updated target image, accordingly.

### Content Loss
The content loss will be the mean squared difference between the target and content features at layer conv4_2. This can be calculated as follows:

content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

### Style Loss
The style loss is calculated in a similar way, only you have to iterate through a number of layers, specified by name in our dictionary style_weights.

Calculate the gram matrix for the target image, target_gram and style image style_gram at each of these layers and compare those gram matrices, calculating the layer_style_loss. Later, this value is normalized by the size of the layer.

### Total Loss
The total loss is derived by adding up the style and content losses and weighting them with your specified alpha and beta!

## Final Outputs

![Image](https://github.com/gdeotale/E4P2/blob/master/Assignment8/StyleTransfer/images/results/all%20outputs%20styple%20transfer.png)

