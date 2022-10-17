

from matplotlib import pyplot as plt
import torch
import torch.utils
import torchvision


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_images(pred, actual):
    pred = pred.reshape(-1, 28, 28).cpu().detach().numpy()
    actual = actual.reshape(-1, 28, 28).cpu().detach().numpy()
    fig, axs = plt.subplots(pred.shape[0], 2)
    for i in range(pred.shape[0]):
        axs[i][0].set_title("Predicted")
        axs[i][0].imshow(pred[i])
        axs[i][1].set_title("Actual")
        axs[i][1].imshow(actual[i])

    plt.show()

BATCH_SIZE = 64

class ClassifierNet(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(ClassifierNet, self).__init__()
        dtype=torch.float32

        self.layer1 = torch.rand(
            size=(input_size, output_size), dtype=dtype, requires_grad=True, device=device)
        self.layer1b = torch.rand(
            size=(output_size,), dtype=dtype, requires_grad=True, device=device)

        self.layers = {
            "layer1": self.layer1,
            "layer1b": self.layer1b,
        }


    def forward(self, x):
        x = torch.mm(x, self.layer1) + self.layer1b
        return x

class ParamTrainerNet(torch.nn.Module):
    def __init__(self, net:ClassifierNet) -> None:
        super(ParamTrainerNet, self).__init__()
        stack = []
        for layerName in net.layers:
            layer = getattr(net, layerName)
            f = torch.nn.Linear(layer.numel(), layer.numel()).to(device)
            stack.append(f)

        self.stack = torch.nn.ModuleList(stack)

    def forward(self, net:ClassifierNet):
        for (i, layerName) in enumerate(net.layers):
            layer = getattr(net, layerName).detach()
            f = self.stack[i]
            out = f(layer.view(-1))
            out = out.view(layer.shape)
            setattr(net, layerName, out)

class SelfLearningModel():
    def __init__(self) -> None:
        pass

    def train(self):
        pass

def to_onehot(args):
    y = torch.zeros((10,), dtype=torch.float32)
    y[args] = 1
    return y

def load_dataset(target_transform=None, batch_size=BATCH_SIZE):
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.flatten(x))
        ]),
        target_transform=target_transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return (train_dataloader,)

def example(dataloader:torch.utils.data.DataLoader):
    BX, BY = next(iter(dataloader))
    BX = BX.to(device)
    BY = BY.to(device)
    classifier = ClassifierNet(BX.shape[1], BY.shape[1]).to(device)
    trainer = ParamTrainerNet(classifier).to(device)

    loss_fn = torch.nn.MSELoss()

    trainer.zero_grad()
    classifier.zero_grad()

    cp1 = get_params(classifier)
    tp1 = get_params(trainer)

    trainer(classifier)
    optimizer = torch.optim.AdamW(params=trainer.parameters(), lr=0.001)

    cp2 = get_params(classifier)
    tp2 = get_params(trainer)
    for i in range(len(cp1)):
        assert torch.any(cp1[i] == cp2[i]) == False
    for i in range(len(tp1)):
        torch.testing.assert_allclose(tp1[i], tp2[i])

    yhat = classifier(BX)
    print(yhat.shape)
    loss = loss_fn(yhat, BY)
    print(loss)

    loss.backward()
    optimizer.step()

    cp4 = get_params(classifier)
    tp4 = get_params(trainer)
    for i in range(len(cp2)):
        # Classifier should not have changed
        torch.testing.assert_allclose(cp2[i], cp4[i])
    for i in range(len(tp2)):
        # Trainer should have changed
        assert torch.all(tp2[i] == tp4[i]) == False
        # assert torch.any(tp2[i] == tp4[i]) == False



def get_params(model:torch.nn.Module):
    return [p.clone().detach() for p in model.parameters()]

def train(classifier:ClassifierNet, trainer:ParamTrainerNet, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.MSELoss, num_epochs:int):
    classifier.train()
    trainer.train()

    for epoch_itr in range(num_epochs):
        for batch_num, (BX, BY) in enumerate(dataloader):
            BX = BX.to(device)
            BY = BY.to(device)

            classifier.zero_grad()
            trainer.zero_grad()

            trainer(classifier)
            optimizer = torch.optim.AdamW(params=trainer.parameters(), lr=0.000001)

            yhat = classifier(BX)
            loss = loss_fn(yhat, BY)
            num_correct = (yhat.argmax(dim=1) == BY.argmax(dim=1)).sum().item()

            loss.backward()
            optimizer.step()

            if batch_num % 100 == 0:
                print("Epoch:", epoch_itr, "Batch:", batch_num, "Loss:", loss.item(), "Correct:", num_correct/BY.shape[0])

        print("Epoch:", epoch_itr, "Loss:", loss.item(), "correct:", num_correct/BY.shape[0])

    print(yhat[:7].argmax(dim=1), "Actual:", BY[:7].argmax(dim=1))
    plot_images(BX[:7], BX[:7])





def main():
    (train_dataloader,) = load_dataset(target_transform=to_onehot)
    example(dataloader=train_dataloader)

    BX, BY = next(iter(train_dataloader))
    classifier = ClassifierNet(BX.shape[1], BY.shape[1]).to(device)
    trainer = ParamTrainerNet(classifier).to(device)
    loss_fn = torch.nn.MSELoss()

    train(classifier, trainer, dataloader=train_dataloader, loss_fn=loss_fn, num_epochs=1)

if __name__ == "__main__":
    main()