from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    names_path = Path(__file__).resolve().parent / "names.txt"
    with names_path.open("r", encoding="utf-8") as file:
        words = [line.strip() for line in file if line.strip()]

    chars = sorted({char for word in words for char in word})

    stoi = {char: index + 1 for index, char in enumerate(chars)}
    stoi["."] = 0
    itos = {index: char for char, index in stoi.items()}

    xs, ys = [], []
    for word in words:
        characters = ["."] + list(word) + ["."]
        for char1, char2 in zip(characters, characters[1:]):
            xs.append(stoi[char1])
            ys.append(stoi[char2])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    generator = torch.Generator().manual_seed(2147483647)
    weight_matrix = torch.randn((len(itos), len(itos)), generator=generator)

    x_encoded = F.one_hot(xs, num_classes=27).float()
    logits = x_encoded @ weight_matrix
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)

    nlls = []
    for i in range(min(5, len(xs))):
        x_index = xs[i].item()
        y_index = ys[i].item()
        x_char = itos[x_index]
        y_char = itos[y_index]
        prob = probs[i, y_index]

        print(f"bigram example {i + 1}: {x_char}{y_char} (indexes {x_index},{y_index})")
        print("input to the neural net:", x_index)
        print("output probabilities from the neural net:", probs[i])
        print("label (actual next character):", y_index)
        print("probability assigned to the correct character:", prob.item())

        log_prob = torch.log(prob)
        nll = -log_prob
        nlls.append(nll)

        print("log likelihood:", log_prob.item())
        print("negative log likelihood:", nll.item())
        print("-" * 30)

    average_nll = torch.stack(nlls).mean().item()
    print("average negative log likelihood, i.e. loss =", average_nll)


if __name__ == "__main__":
    main()
