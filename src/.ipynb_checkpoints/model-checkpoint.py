import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        
        super(MyModel, self).__init__()


        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        # Define the CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Convolutional layer 1
            nn.ReLU(inplace=True),                      # ReLU activation 1
            nn.MaxPool2d(kernel_size=2, stride=2),      # Max pooling 1

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Convolutional layer 2
            nn.ReLU(inplace=True),                       # ReLU activation 2
            nn.MaxPool2d(kernel_size=2, stride=2),       # Max pooling 2

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Convolutional layer 3
            nn.ReLU(inplace=True),                        # ReLU activation 3
            nn.MaxPool2d(kernel_size=2, stride=2),        # Max pooling 3

            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Convolutional layer 4
            nn.ReLU(inplace=True),                        # ReLU activation 4
            nn.MaxPool2d(kernel_size=2, stride=2)         # Max pooling 4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),                       # Dropout layer
            nn.Linear(512 * 14 * 14, 4096),              # Fully connected layer 1
            nn.ReLU(inplace=True),                       # ReLU activation
            nn.Dropout(p=dropout),                       # Dropout layer
            nn.Linear(4096, 4096),                       # Fully connected layer 2
            nn.ReLU(inplace=True),                       # ReLU activation
            nn.Linear(4096, num_classes)                 # Fully connected layer 3
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        # Pass the input through the feature extractor
        x = self.features(x)
        # Flatten the output tensor
        x = x.view(x.size(0), -1)
        # Pass the flattened tensor through the classifier
        x = self.classifier(x)
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
