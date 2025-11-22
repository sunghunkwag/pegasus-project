# import torch # Not available in this environment
# import torch.nn.functional as F
# from model import HRM
# from data_loader import SimpleTokenizer

# Mocking necessary components for simulation in an environment without PyTorch
class MockTensor:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    def unsqueeze(self, dim):
        return self

class MockTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

class MockHRM:
    def __init__(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self

    def __call__(self, x, skill_vec=None):
        # This mock simulates the behavior of the HRM
        # It checks the skill_vec to determine which 'persona' is active
        # skill_vec is expected to be a MockTensor wrapping a list/array

        skill = skill_vec.data[0] # Get the vector for the first item in batch

        # Simple logic to detect which one-hot index is active
        if skill[0] == 1:
            return "sarcastic", None
        elif skill[1] == 1:
            return "wise", None
        else:
            return "default", None

def run_inference_simulation():
    print("Initializing Simulation (Mocked Environment)...")

    # Setup
    tokenizer = MockTokenizer()

    # Instantiate Model (Mocked)
    hrm = MockHRM()
    hrm.eval()

    # Prompt
    prompt = "What is death?"
    print(f"\nPrompt: {prompt}")
    prompt_tokens = MockTensor(tokenizer.encode(prompt))

    # ---------------------------------------------------------
    # Simulation: Persona A (Skill [1, 0, 0, 0]) - Sarcastic
    # ---------------------------------------------------------
    # Simulating: skill_vec_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).to(device)
    skill_vec_a = MockTensor([[1, 0, 0, 0]])

    output_type, _ = hrm(prompt_tokens, skill_vec=skill_vec_a)

    print("\n--- Persona A (Skill: [1, 0, 0, 0]) ---")
    print("System: Modulating High-Level State h_h with Skill Vector A...")

    if output_type == "sarcastic":
        print("Output (Simulated):")
        print("> \"Death is nature's way of telling you to stop talking. It's the ultimate deadline, literally.\"")

    # ---------------------------------------------------------
    # Simulation: Persona B (Skill [0, 1, 0, 0]) - Wise
    # ---------------------------------------------------------
    # Simulating: skill_vec_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).to(device)
    skill_vec_b = MockTensor([[0, 1, 0, 0]])

    output_type, _ = hrm(prompt_tokens, skill_vec=skill_vec_b)

    print("\n--- Persona B (Skill: [0, 1, 0, 0]) ---")
    print("System: Modulating High-Level State h_h with Skill Vector B...")

    if output_type == "wise":
        print("Output (Simulated):")
        print("> \"Death is not an end, but a transformation. It is the silence that gives meaning to the song of life.\"")

if __name__ == "__main__":
    run_inference_simulation()
