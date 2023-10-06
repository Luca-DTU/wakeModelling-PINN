import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to apply the initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Instantiate the network
net = Net()

print("Before Initialization:")
for param in net.parameters():
    print(param)

# Apply the initialization
net.apply(weights_init)

# Print model parameters after initialization
print("\nAfter Initialization:")
for param in net.parameters():
    print(param)

# Create a sample input tensor
rz_star = torch.tensor([[0.5, 0.5]], requires_grad=True)

# Pass the input through the network
uvp_star = net(rz_star)

# Print the output and grad_fn
print(uvp_star)
print(uvp_star.grad_fn)

# Post-process the output (analogous to your denormalization)
uvp_star = uvp_star * 2.0
# with torch.no_grad():
#     rz_star = rz_star*2.0
# rz_star.requires_grad_(True)

# Print the post-processed output and grad_fn
print(uvp_star)
print(uvp_star.grad_fn)

# Extract one component of the output
u_r_star = uvp_star[:, 0]

# Compute the gradient of that component with respect to the input
du_r_star = torch.autograd.grad(u_r_star, rz_star, grad_outputs=torch.ones_like(u_r_star), create_graph=True)[0]

# Print the computed gradient
print(du_r_star)
