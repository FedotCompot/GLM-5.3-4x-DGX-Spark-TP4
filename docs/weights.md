# Weights and immutable manifests

Every node stores a full target checkpoint at identical absolute paths. Native MTP is included in
that checkpoint. The unmodified Red Hat drafter is required only for the DSpark K3 alternative.
TP4 shards the loaded model, not the files on disk. No weights are included in this repository.

| checkpoint | revision | safetensors files | total downloaded files |
|---|---|---:|---:|
| Tech2wild/GLM-5.3-Int4-Int8Mix | `206507bbb047d8223964a0414cd83230c59428f9` | 282 | 290 |
| RedHatAI/GLM-5.3-speculator.dspark | `b374b95663447ea0e935151be4f3d6666e36e6d7` | 1 | 6 |

`manifests/target.json` and `drafter.json` record full revisions, file names, byte sizes and SHA256.
The companion `.sha256` files use the standard sha256sum format. The target is about 405.28 GB and
the drafter about 2.50 GB. `.gitattributes` and the local backup template are not serving assets and are excluded.

```bash
scripts/download_weights.sh "$HOME/models"
python3 scripts/weights.py verify target "$HOME/models/Tech2wild/GLM-5.3-Int4-Int8Mix"
# Only for RECIPE_PROFILE=dspark-k3:
python3 scripts/weights.py download drafter "$HOME/models/RedHatAI/GLM-5.3-speculator.dspark"
python3 scripts/weights.py verify drafter "$HOME/models/RedHatAI/GLM-5.3-speculator.dspark"
```

The downloader uses public immutable resolve URLs, streams to `.part`, resumes with Range requests,
checks size/SHA256 before rename and refuses a mismatched existing final file. It does not need a token
or import a GPU framework. Large-file hashes come from the saved immutable checkpoint manifests;
small-file hashes were completed from read-only deployed-file reads. The target's large-file manifest
was verified on all four ranks in Boot B. This packaging job did not rehash 1.6 TB of target weights.

`./start.sh preflight` and `serve` hash every file on the first run on each host and keep a receipt in
`~/.cache/glm53-full-verify/`, keyed by the manifest and directory, with each file's size, mtime, ctime,
inode and device. Later runs skip hashing only if all of these are unchanged; any change rehashes every
file. A receipt older than seven days is ignored, so every host rehashes everything at least once a week.
`VERIFY_WEIGHTS=full ./start.sh serve` always hashes everything and does not use the receipt.

The pinned target template already supports `enable_thinking=false` by appending an empty think block
at generation time; no template patch is needed. Its SHA256 is
`9687489b1c7eddac80c78c7d57a11e791601013c46dfd7202241fed85e340a74`, matching the Boot B saved template.
The target config and drafter config were checked against their pinned public resolve URLs.

The target and RedHat drafter use the GLM-5.3 model licence; see their pinned model cards and the
[target licence](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/blob/206507bbb047d8223964a0414cd83230c59428f9/LICENSE).
The recipe's code licence does not grant rights to model weights. No drafter fine-tune, head conversion,
Markov modification or quantization is applied. Incoai DFlash2 is CC BY-NC-ND 4.0 and is not downloaded
or used by this recipe. CPU-only allocator fixtures exercise alternate synthetic layouts; they do not
install alternative models or select alternative serving profiles.

## Abliterated alternative (`RECIPE_MODEL=abliterated`)

| checkpoint | revision | safetensors files | total downloaded files |
|---|---|---:|---:|
| Blackfrost-AI/GLM-5.3-DERISKED-Int4-Int8Mix | `08a92f9da01cd90d591edb3fdd71c2ccc9f37ddd` | 282 | 290 |

Public and ungated, GLM-5.3 licence. Per its model card, Blackfrost copied the pinned Tech2wild tree (revision
`206507bb`) and applied one direction-weight modification pass (alpha 3.0, first two layers skipped, scales
frozen) to 76 `o_proj`, 1 dense `down_proj`, 19,200 routed-expert `down_proj` and 75 shared `down_proj`
matrices; the MTP layer is stock. It is the public checkpoint matching the "abliterated graft with the identical
quantization envelope" that ajclark's DCP4 recipe reports measuring; that recipe does not name its weights, so
this identity is not confirmed. The card states it has not been through a judged refusal suite.

`manifests/abliterated.json` was generated with `scripts/weights.py manifest` (Hub LFS SHA256 for shards, local
SHA256 for small files); the same command reproduces `manifests/target.json` byte for byte. Against the stock
target, all 290 file names, `config.json`, `model.safetensors.index.json`, the chat template, tokenizer and
generation config are identical; 278 of 282 shards differ (re-serialised), and shards 1 and 271-273 are identical.
Other public abliterated GLM-5.3 checkpoints use NVFP4, FP8 or BF16 layouts this recipe does not load.

```bash
scripts/download_weights.sh "$HOME/models" abliterated
python3 scripts/weights.py verify abliterated "$HOME/models/Blackfrost-AI/GLM-5.3-DERISKED-Int4-Int8Mix"
# Convert both NVFP4 sidecars FROM THIS TREE into new directories (docs/NVFP4-SIDECARS.md, MODEL_DIR=<this tree>).
# .env, next to the unchanged stock paths:
#   RECIPE_MODEL=abliterated
#   ABLITERATED_MODEL_DIR=/srv/glm/models/Blackfrost-AI/GLM-5.3-DERISKED-Int4-Int8Mix
#   ABLITERATED_ATTN_NVFP4_DIR=/srv/glm/models/GLM-5.3-derisked-attn-nvfp4
#   ABLITERATED_NVFP4_MORE_DIR=/srv/glm/models/GLM-5.3-derisked-more-nvfp4
DRY=1 ./start.sh serve   # prints "Model: abliterated (Blackfrost-AI/...@08a92f9...)"
./start.sh preflight
```

`RECIPE_MODEL=stock` (default) returns to the Tech2wild tree; a switch is a drained restart. The served name stays
`GLM-5.3`, so clients and bench tools need no change; DRY, serve and `state/deployment.json` record the model.

**Sidecars must come from the selected tree.** The NVFP4 sidecars replace every attention projection
(including `o_proj`), the shared experts and the dense MLPs. Sidecars converted from the stock tree would load
silently over the abliterated checkpoint, because the loaders only compare `config.json` and the index, which are
byte-identical, and would restore stock `o_proj` and shared/dense `down_proj` weights. `./start.sh preflight`
therefore compares each mounted sidecar's recorded source shard hashes with the selected manifest and refuses a
mismatch. Convert both sidecars from the abliterated tree (about 7.5 + 5 CPU minutes per rank, measured on the stock tree).

Unqualified on this recipe: no boot, memory, quality, needle, temperature-0 or speed evidence exists for the
abliterated tree; the release-gate results are stock-only. The native MTP draft layer is stock, so draft
acceptance (and K-stop's stock-fitted tau) may move; verification keeps outputs exact to the abliterated target.
The model card's disclaimer applies: the altered refusal profile is not a safety boundary, and operators are
responsible for lawful use.
