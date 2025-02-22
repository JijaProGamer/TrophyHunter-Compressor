import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MapSimplifier(nn.Module):
    def __init__(self, input_size, map_size, num_blocks):
        super(MapSimplifier, self).__init__()
        self.map_size = map_size
        self.num_blocks = num_blocks
        self.height, self.width = map_size

        out_size = self.height * self.width * num_blocks

        self.dense1 = nn.Linear(input_size, 2048, bias=False)
        self.bn1 = nn.BatchNorm1d(2048)

        self.dense2 = nn.Linear(2048, 1024, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)

        self.dense3 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)

        self.dense4 = nn.Linear(512, out_size, bias=False)
        self.bn4 = nn.BatchNorm1d(out_size)
    def forward(self, x):
        x = F.silu(self.bn1(self.dense1(x)))
        x = F.silu(self.bn2(self.dense2(x)))
        x = F.silu(self.bn3(self.dense3(x)))
        x = F.silu(self.bn4(self.dense4(x)))
        
        return x.view(-1, self.height, self.width, self.num_blocks)


class MapGeneratorWrapper:
    def __init__(self, input_size, map_size, num_blocks, device):
        self.model = MapSimplifier(input_size, map_size, num_blocks).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            predicted_blocks = torch.argmax(output, dim=-1)
            
            return predicted_blocks

    def loss(self, output, targets):
        return self.criterion(output.permute(0, 3, 1, 2).reshape(output.shape[0], self.model.num_blocks, -1), targets.view(targets.shape[0], -1))

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.loss(self.model(x), y)

        loss.backward()
        self.optimizer.step()

        return loss.item()


batch_size = 256
input_size = 2048
map_size = (24, 15)
num_blocks = 6
# air, block, indestructible, grass, water, oneshot

device = torch.device("cuda")


x_train = torch.randn(batch_size, input_size).to(device)

y_train = torch.randint(0, num_blocks, (batch_size, map_size[0], map_size[1])).to(device)

map_generator = MapGeneratorWrapper(input_size, map_size, num_blocks, device)

print("Training...")
for epoch in range(300):
    loss = map_generator.train_step(x_train, y_train)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


print("\nTesting Inference...")
x_test = torch.randn(1, input_size).to(device)
predicted_map = map_generator.predict(x_test)

print("Predicted Map Shape:", predicted_map.shape)
print(predicted_map)
