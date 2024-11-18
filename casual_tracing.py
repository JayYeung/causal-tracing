import os, re, json
import matplotlib.pyplot as plt
import torch, numpy
from tqdm import trange, tqdm
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from dsets.mquake import MQuAKEPromptCompletionDataset

torch.set_grad_enabled(False)

model_name = '/data/akshat/models/gpt2-xl'
model_name = "/data/akshat/models/Llama-2-7b-hf"


model_output_name = model_name.split('/')[-1]

mt = ModelAndTokenizer(
    model_name,
)

print("The Space Needle is in the city of", predict_token(
    mt,
    ["The Space Needle is in the city of"],
    return_p=True,
))


dataset = MQuAKEPromptCompletionDataset(max_examples=300)
subjects = [item['prompt'] for item in dataset]
noise_level = 3 * collect_embedding_std(mt, subjects)
print(f"Computed noise level: {noise_level}")

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf = f"output/casual_trace_{kind}", modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    for kind in [None, "mlp", "attn"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind
        )

def aggregate_probabilities_by_layer(mt, prompts, answers, noise_level, num_layers):
    layer_probs = []
    invalid_cached = set()

    for layer in trange(num_layers):
        layer_sum = 0
        valid_count = 0  # Track number of valid predictions
        for prompt, correct_answer in zip(prompts, answers):
            if prompt in invalid_cached:
                continue
            
            correct_answer_token = mt.tokenizer.encode(correct_answer, return_tensors='pt').cuda()[0]
            correct_answer_token_w_space = mt.tokenizer.encode(' ' + correct_answer, return_tensors='pt').cuda()[0]

            correct_answer_token_id = int(correct_answer_token[0].item())
            correct_answer_token_w_space_id = int(correct_answer_token_w_space[0].item())

            inp = make_inputs(mt.tokenizer, [prompt] * 2)  
            answer_t, _ = [d[0] for d in predict_from_input(mt.model, inp)]
            
            guess_text = decode_tokens(mt.tokenizer, [answer_t])[0]

            if int(answer_t) not in {correct_answer_token_id, correct_answer_token_w_space_id}:
                invalid_cached.add(prompt)
                continue
                
            # print(f'prompt: {prompt}, guess: {repr(guess_text)}, correct: {repr(correct_answer)}')


            last_token_position = len(inp["input_ids"][0]) - 3
            tokens_to_mix = (last_token_position, last_token_position + 3)
            # print(f'e_range: {tokens_to_mix}')
            
            prob = trace_with_patch(
                mt.model,
                inp,
                [(len(inp["input_ids"][0]) - 1, layername(mt.model, layer))],
                answer_t,
                tokens_to_mix=tokens_to_mix,
                noise=noise_level,
            ).item()
            

            layer_sum += prob
            valid_count += 1
        print(f'Layer {layer}: {layer_sum / valid_count} ({valid_count} valid predictions)')
        if valid_count > 0:
            layer_probs.append(layer_sum / valid_count)
        else:
            layer_probs.append(0) 

    return layer_probs


def plot_layer_probabilities(layers, probabilities, output_dir="output", filename="layer_probabilities.png"):
    os.makedirs(output_dir, exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(layers, probabilities, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Aggregated Probability")
    plt.title("Aggregated Probabilities by Layer")
    plt.grid()

    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

prompts = [item['prompt'] for item in dataset]
answers = [item['answer'] for item in dataset]
num_layers = mt.num_layers
layers = list(range(num_layers))

aggregated_probs = aggregate_probabilities_by_layer(mt, prompts, answers, noise_level, num_layers)
plot_layer_probabilities(layers, aggregated_probs, output_dir="output", filename=f"layer_probabilities_{model_output_name}.png")