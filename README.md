# Oblique abliteration

Abliteration removes a model's refusal behaviour. It also, as a side effect
nobody chose, damages the model's willingness to say "I don't know", so the
model fabricates instead of declining. This fixes that side effect by changing
one vector in the projector.

```
standard:   y' = y - w (r . y) r
this:       y' = y - w (u . y) r        where  u . r = 1,  u . j = 0
```

`y` is what a block writes into the residual stream and `w` is how hard to cut.
Three vectors, and the change is which one does the measuring:

- **`r`** is the refusal direction. It is what gets subtracted, in both versions.
- **`j`** is the direction to protect. Here it is hedging: the model's
  willingness to answer "that is not in the context" instead of guessing.
- **`u`** is the ruler, the vector dotted against the activation to decide how
  much to subtract. Standard abliteration uses `r` for this as well. This uses a
  separate `u`, built so that `u . j = 0` and `u . r = 1`.

Because `u . j = 0`, the edit reads zero of `j` and so cannot touch it. Because
`u . r = 1`, refusal is still removed at full strength.

Measured on Qwen3-4B, holding refusal removal equal at 98% or better: the
standard edit kept **71%** of the model's original hedging behaviour, the
oblique edit kept **96%**.

---

## Run it

```bash
git clone git@github.com:pigeondrops/obliteration.git
cd obliteration
pip install -e .
python scripts/fetch_data.py
```

`--model` takes either form:

```bash
--model Qwen/Qwen3-4B                      # repo id, downloaded and cached
--model /path/to/local/model               # a directory you already have
```

A local directory needs the usual HuggingFace layout: `config.json`, the
`*.safetensors` shards with their `model.safetensors.index.json`, and the
tokenizer files. That is exactly what a `snapshot_download`, a `git clone` of a
model repo, or a previous run's `--out` directory gives you, so an already
edited model can be fed straight back in.

Check separability first. One model load, no search, tells you whether the
method can help on your model at all:

```bash
oblique-abliterate --model Qwen/Qwen3-4B --preserve-grounding --dry-run
```

```
[oblique] Qwen3-4B-oblique | fit 64+64 | eval 48+48
[oblique] grounding contrast: 64 unanswerable / 64 answerable
[oblique] 36 decoder layers
[oblique] |cos(refusal, hedging)| median over upper half = 0.170
          (low means separable and the projector is well conditioned)
[oblique] dry run: geometry only, stopping before search
```

The same thing against a local directory:

```bash
oblique-abliterate \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<hash> \
    --preserve-grounding --dry-run
```

Then run it:

```bash
oblique-abliterate --model Qwen/Qwen3-4B --preserve-grounding \
    --out out/qwen3-4b-oblique
```

That is the whole command. `--device cpu`, `--trials 40` and `--n 256` are the
defaults and produced every result below, so there is nothing else to set for a
first run. It writes the edited model, an `abliteration.json` manifest recording
exactly what was done, and a study file with every trial.

Replay a previous run without re-searching:

```bash
oblique-abliterate --model Qwen/Qwen3-4B --preserve-grounding \
    --study out/qwen3-4b-oblique.study.json --export-from-study
```

### Flags that matter

| flag | why |
| --- | --- |
| `--preserve-grounding` | the point of this package. Without it you get ordinary abliteration, which is only useful as a baseline. |
| `--device cpu` | default. The model is held in bf16 for the edit, so a 27B needs ~54 GB RAM and most GPUs cannot fit one. |
| `--trials` | more trials, better configs found. 40 produced every result below. |
| `--kl-target` | stop optimising damage below this and spend the rest of the budget on refusals. Default 0.01. |
| `--preserve-rank` | keep a subspace instead of one direction. Untested, see limits. |

---

## How it works

### The projector

Using `r` as its own ruler makes the projection symmetric, which protects
exactly the orthogonal complement of `r`. Nobody chose that subspace, and it is
usually not the thing worth keeping.

What it fails to protect is measurable. Refusal and hedging are distinct
directions but not perpendicular: on Qwen3-4B layer 21 the cosine between them
is 0.18. Measuring with `r` therefore picks up 18% of whatever hedging is
present and subtracts that too. The model stops refusing, and stops admitting
ignorance along with it.

Building `u` removes that leak:

```
u0 = r - (j . r) j      strip the part of r that lies along j
u  = u0 / (u0 . r)      rescale so u reads exactly one unit of r
```

Both properties then hold exactly, not approximately, and both are asserted
numerically in `tests/test_projector.py`.

**Cost.** The edit is amplified by `||u||/||r||`, bounded by `1/sin(angle)`
between `r` and `j`. At cosine 0.18 that is 1.017, effectively free. It grows
without limit as the behaviours approach parallel; the code reports it as
`oblique_leverage`. If `r` lies inside the span of `j`, no such projector
exists and the code raises instead of returning something unusable. That case is
worth knowing about: it says the two behaviours are not linearly separable at
that layer.

### The pipeline

1. **Extract.** Mean last-token hidden state per layer for each prompt set,
   then subtract. Last token because that is the position the model generates
   from, so it carries the decision about how to answer.
2. **Orthogonalise.** Remove the component of the refusal direction lying along
   the harmless representation, so the edit does not drag general capability
   along with refusal.
3. **Search.** Nine parameters: a continuous direction index (15.81 means 19% of
   the way from layer 15's direction to layer 16's) plus four shaping a
   triangular depth kernel, fitted separately for attention and MLP.
   Two-objective TPE minimising refusals and KL jointly.
4. **Score.** Damage is KL divergence on first-token log-probabilities against
   the unmodified model, rather than a judgement about generated text. Text
   graders drift as they are tuned, miss refusals phrased unusually, and count a
   truncated answer as a failure. KL has no grader, so none of that can happen.
5. **Export.** Apply the winning config to real weights once.

The search bounds carry the assumption that refusal sits past the midpoint of
the stack. They matter: sampling the full depth range lets the kernel peak at
layer 0, which produced KL between 1.0 and 2.5, a destroyed model, where the
bounded search reaches KL well under 0.1.

### Why forward hooks, and why on blocks

Hundreds of candidates have to be tried, and writing weights each time would be
unusably slow. Hooks are not an approximation. For `y = Wx`:

```
(W - w r u^T W) x  ==  y - w (u . y) r
```

Hooking the **block** rather than individual projections is what reaches a
fused-expert MoE. On the Qwen3.6 layout, `mlp.experts.down_proj` is a single 3-D
Parameter holding all experts at once rather than one Module per expert, so
there is no submodule to hook. Projection is linear, so:

```
P(sum_e down_proj_e(x) + shared(x)) == sum_e P(down_proj_e(x)) + P(shared(x))
```

One hook per block covers every expert. The export then edits the 3-D stack
directly, expert by expert.

On the MoE I measured, the refusal write splits **4.6% attention, 43.5% shared
expert, 52% fused routed experts**, so most of it lives in that stack.

---

## Models tested

Everything here is Qwen. None of it is a general claim about language models.

| model | role | result |
| --- | --- | --- |
| Qwen3-0.6B | smallest worked example, runs on CPU | geometry only |
| Qwen3-4B | all geometry, and the per-lever table below | hedging 71% to 96% |
| Qwen3.6-35B-A3B | fused-expert MoE, shipped | 0/40 refusals, KL 0.276 |
| Qwen3.8-27B | dense, shipped | 2/40 from 40/40, KL 0.060 |

Per-lever on Qwen3-4B, at matched refusal removal of 98% or better:

| variant | grounded hedging retained |
| --- | --- |
| standard (orthogonal projection) | 71% |
| oblique, which is what this package does | 96% |
| oblique plus two further refinements | 98%, KL 0.02 |

The last row is listed for context and **is not reachable with this package**.
It adds per-column grading and per-block attribution, neither of which is
implemented here. Per-column grading in particular is not the same operation on
a fused expert stack, where `down_proj` is one 3-D tensor rather than a matrix
per expert, so it would need validating separately. What ships here is the
middle row.

**Separability**, what `--dry-run` reports. Low means two genuinely distinct
behaviours and a nearly free edit; high means entangled, and no projector can
remove one while keeping the other.

| model | median abs cos, upper half | n | leverage |
| --- | --- | --- | --- |
| Qwen3-0.6B | 0.083 | 24 | 1.004 |
| Qwen3-4B | 0.170 | 64 | 1.015 |
| Qwen3.8-27B | 0.041 | 64 | 1.001 |

Leverage is `1/sin(angle)`, the factor the edit is amplified by. At these
cosines it is under 1.02, so the oblique edit costs essentially nothing. The
35B-A3B is absent because its separability has not been measured.

These are medians across the upper half of the stack, at the `n` shown, so the
rows are not directly comparable with each other. The 0.18 used in the leverage
example earlier is a single layer, Qwen3-4B layer 21, which is a different
quantity again.

**Not tested:** Llama, Mistral, Gemma, DeepSeek. No non-Qwen architecture, in
any size. Every figure above comes from a single seed, one model per regime, at
the sample sizes shown.

**Why it may not transfer:** the writer-selection filter matches on module
*name* and tensor *shape*, so it is architecture-specific by construction. If
nothing matches, the run errors rather than writing an unmodified copy. The
fused 3-D expert stack is a Qwen3.6 layout detail; other MoEs give each expert
its own module. The depth band where refusal concentrates is a per-model
observation, not a constant.

Run `--dry-run` on your model before assuming any of it carries over.

---

## Data

Four prompt sets, in two matched pairs:

```
harmful / harmless          ->  the refusal direction r
unanswerable / answerable   ->  the direction to preserve j
```

**Matching is not a detail.** Everything the two sides share cancels in the
difference of means, so matched sets isolate the behaviour and unmatched sets
capture dataset identity instead. With an unmatched pair I measured roughly a 35
degree rotation away from the true refusal direction, enough to make the whole
edit land in the wrong place. Keep both sides the same size.

**Grounding pairs ship with the package** as `data/*.jsonl`, generated by
`obliteration/data.py`. Each pair shares a byte-identical three-fact context
about an invented entity and differs only in whether the queried fact is
present:

```
Context: They hold their festival on Highmoon. Their staple food is Litei
root. Their homeworld is Movoo.

Question: What is the homeworld of the Kinaguok?     <- answerable
Question: Who rules the Kinaguok?                    <- unanswerable
```

The entities are nonsense on purpose. A real one would let the model answer from
parametric memory instead of from the context, which would measure recall rather
than grounding.

They are JSONL rather than plain text because each prompt contains a blank line,
so one prompt per line would split every record in two. `write_prompts` checks
the round-trip before writing.

**The refusal pair is not redistributed here.** `scripts/fetch_data.py` pulls it
from the original sources:

- **harmful** comes from **AdvBench**, released with *Universal and Transferable
  Adversarial Attacks on Aligned Language Models*, Zou, Wang, Carlini et al.,
  [arXiv:2307.15043](https://arxiv.org/abs/2307.15043). Repository:
  [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks), MIT.
  The script reads `data/advbench/harmful_behaviors.csv`.
- **harmless** comes from **Stanford Alpaca**, Taori, Gulrajani, Zhang et al.,
  [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca).
  The code is Apache-2.0 but the dataset is **CC BY-NC 4.0**, so it is
  non-commercial and requires attribution. The script keeps only the
  no-input instructions, so the prompt shape matches AdvBench's.

Neither set is vendored. Both are fetched from the sources above at the version
those repositories currently serve, so the licences stay with their owners and
there is no stale copy here to drift out of date.

If you point `--harmful` and `--harmless` at your own sets instead, the licences
above stop applying and yours do.

---

## What is new here, and what is not

**Not new:** the refusal direction, difference-of-means extraction, using TPE to
search edit parameters, or abliterating a fused-stack MoE by direct weight
editing. All of that is established practice.

**New:** the oblique projector applied to abliteration, separating the removal
direction from the measurement covector so a named second behaviour is preserved
exactly rather than incidentally. Plus the measured attribution of the refusal
write on a fused MoE, 4.6 / 43.5 / 52.

## Limits

- Refusal counting is substring matching. It undercounts unusual phrasings and
  cannot see an evasive non-refusal. It is a search signal, not a benchmark.
- KL is first-token only, over harmless prompts, so it says nothing about
  long-form degradation.
- `--preserve-rank > 1` uses neighbouring layers' directions as a proxy basis.
  Rank 1 produced every number above.
- Row normalisation, present in some other implementations, is not applied.
- Requantising an abliterated model can move behaviour. Re-run your grounding
  checks afterwards.

## Prior work

This builds directly on three pieces of earlier work.

- **Arditi, Obeso, Syed, Paleka et al.**, *Refusal in Language Models Is
  Mediated by a Single Direction*,
  [arXiv:2406.11717](https://arxiv.org/abs/2406.11717). The foundation: a single
  direction carries refusal, and it sits slightly past the midpoint of the
  stack. The search bounds here encode that finding.
- **Heretic**, [p-e-w/heretic](https://github.com/p-e-w/heretic). Searching
  abliteration parameters with TPE, scored jointly on refusal count and KL. The
  objective used here is the same one. No Heretic code is used, copied or
  adapted; what is shared is the approach, not an implementation. Heretic is
  AGPL-3.0 and this repository is MIT, so read both licences before combining
  them.
- **grimjim** (Jim Lai),
  [huggingface.co/grimjim](https://huggingface.co/grimjim). Orthogonalising the
  refusal direction against the harmless direction before the edit, which
  `build_directions(project=True)` implements.

## Responsible use

This removes safety refusals from a language model. It is published because the
technique is already public, and because the grounding result is worth having in
the open: a model that confidently invents an answer does more harm than one
that declines to give it.

If you publish a model made with this, say that it was abliterated and ship the
`abliteration.json` manifest alongside the weights. It records the direction,
the depth curve and the strengths used, which is the only way anyone downstream
can tell what was changed and by how much. Say what you measured too, including
whether grounding was checked, since that is the failure this method exists to
avoid and the one least likely to be noticed.

## License

MIT. See `LICENSE`.
