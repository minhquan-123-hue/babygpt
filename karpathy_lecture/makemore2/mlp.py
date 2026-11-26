from pathlib import Path

import torch
import torch.nn.functional as F


NAMES_PATH = Path(__file__).parent.parent / "makemore" / "names.txt"
BLOCK_SIZE = 3
EMBEDDING_SIZE = 2
HIDDEN_SIZE = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.1
TRAINING_STEPS = 9_000
SEED = 2_147_483_647


def build_dataset(
    words: list[str], stoi: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []

    for word in words:
        context = [0] * BLOCK_SIZE
        for character in word + ".":
            index = stoi[character]
            inputs.append(context)
            targets.append(index)
            context = context[1:] + [index]

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def split_words(
    words: list[str], generator: torch.Generator
) -> tuple[list[str], list[str], list[str]]:
    shuffled_words = words.copy()
    shuffled_indices = torch.randperm(len(shuffled_words), generator=generator).tolist()
    shuffled_words = [shuffled_words[index] for index in shuffled_indices]

    training_end = int(len(shuffled_words) * 0.8)
    validation_end = int(len(shuffled_words) * 0.9)
    return (
        shuffled_words[:training_end],
        shuffled_words[training_end:validation_end],
        shuffled_words[validation_end:],
    )


def main() -> None:
    words = NAMES_PATH.read_text(encoding="utf-8").splitlines()
    characters = sorted(set("".join(words)))
    stoi = {character: index + 1 for index, character in enumerate(characters)}
    stoi["."] = 0

    generator = torch.Generator().manual_seed(SEED)
    training_words, validation_words, test_words = split_words(words, generator)
    training_inputs, training_targets = build_dataset(training_words, stoi)
    validation_inputs, validation_targets = build_dataset(validation_words, stoi)
    test_inputs, test_targets = build_dataset(test_words, stoi)

    embedding = torch.randn((len(stoi), EMBEDDING_SIZE), generator=generator)
    w1 = torch.randn((BLOCK_SIZE * EMBEDDING_SIZE, HIDDEN_SIZE), generator=generator)
    b1 = torch.randn(HIDDEN_SIZE, generator=generator)
    w2 = torch.randn((HIDDEN_SIZE, len(stoi)), generator=generator)
    b2 = torch.randn(len(stoi), generator=generator)
    parameters = [embedding, w1, b1, w2, b2]

    print(sum(parameter.numel() for parameter in parameters))
    for parameter in parameters:
        parameter.requires_grad = True

    batch_size = min(BATCH_SIZE, training_inputs.shape[0])

    for _ in range(TRAINING_STEPS):
        batch_indices = torch.randperm(
            training_inputs.shape[0], generator=generator
        )[:batch_size]
        batch_inputs = training_inputs[batch_indices]
        batch_targets = training_targets[batch_indices]

        embedded = embedding[batch_inputs]
        hidden = torch.tanh(
            embedded.reshape(batch_inputs.shape[0], BLOCK_SIZE * EMBEDDING_SIZE) @ w1 + b1
        )
        logits = hidden @ w2 + b2
        loss = F.cross_entropy(logits, batch_targets)

        for parameter in parameters:
            parameter.grad = None
        loss.backward()

        with torch.no_grad():
            for parameter in parameters:
                parameter -= LEARNING_RATE * parameter.grad

    with torch.no_grad():
        batch_embedded = embedding[batch_inputs]
        batch_hidden = torch.tanh(
            batch_embedded.reshape(
                batch_inputs.shape[0], BLOCK_SIZE * EMBEDDING_SIZE
            )
            @ w1
            + b1
        )
        batch_logits = batch_hidden @ w2 + b2
        last_batch_loss = F.cross_entropy(batch_logits, batch_targets)

        validation_embedded = embedding[validation_inputs]
        validation_hidden = torch.tanh(
            validation_embedded.reshape(
                validation_inputs.shape[0], BLOCK_SIZE * EMBEDDING_SIZE
            )
            @ w1
            + b1
        )
        validation_logits = validation_hidden @ w2 + b2
        validation_loss = F.cross_entropy(validation_logits, validation_targets)

        test_embedded = embedding[test_inputs]
        test_hidden = torch.tanh(
            test_embedded.reshape(test_inputs.shape[0], BLOCK_SIZE * EMBEDDING_SIZE)
            @ w1
            + b1
        )
        test_logits = test_hidden @ w2 + b2
        test_loss = F.cross_entropy(test_logits, test_targets)

    print(
        f"Dataset split: {len(training_words)} training words, "
        f"{len(validation_words)} validation words, {len(test_words)} test words"
    )
    print(
        f"Last training batch loss ({batch_size} examples): "
        f"{last_batch_loss.item():.4f}"
    )
    print(
        f"Validation loss ({validation_inputs.shape[0]} examples): "
        f"{validation_loss.item():.4f}"
    )
    print(f"Test loss ({test_inputs.shape[0]} examples): {test_loss.item():.4f}")


if __name__ == "__main__":
    main()
