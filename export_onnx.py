import torch


class PolicyForExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        means = out[:, :8]
        actions = torch.tanh(means)  # Clamp actions to [-1, 1]
        return actions


def export(policy, obs_size: int = 89, output_path: str = "model.onnx"):
    policy.eval()
    export_model = PolicyForExport(policy)
    export_model.eval()

    dummy_input = torch.zeros(1, obs_size)

    # Debug: show raw vs processed outputs before export
    with torch.no_grad():
        raw_out = policy(dummy_input)
        print(f"Raw Output (all 16): {raw_out}")
        print(f"Raw Means (0-7): {raw_out[:, :8]}")
        print(f"Raw LogStds (8-15): {raw_out[:, 8:]}")
        processed = export_model(dummy_input)
        print(f"Exported Actions (tanh applied): {processed}")

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        opset_version=11,
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
    )
    print(f"Model exported to {output_path}")
