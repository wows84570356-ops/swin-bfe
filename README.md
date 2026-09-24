# Swin-BFE: building footprint extraction with an encoder-weighted design

Code and per-image records for *Building Footprint Extraction With an
Encoder-Weighted Swin Transformer and Multi-Granularity Analysis*.

The network puts 98.4% of its parameters in a pretrained Swin-Small encoder and
0.81 M in everything after it: no bottleneck stage, four 1x1 laterals to a
common width of 96, parameter-free bilinear upsampling, addition rather than
concatenation, and a mask head that stops at stride 4 and interpolates once to
full resolution. The reference architecture it is compared against distributes
capacity the other way, 7.1% in its encoder and 92.9% after it.

Three interchangeable blocks occupy the context slot, spanning 0.21 to 0.50 M
parameters. On WHU they leave pixel IoU, Boundary IoU and the spurious-component
rate within the retraining noise floor, so the result does not depend on a
particular decoder module. A ResNet-34 U-Net that also holds 85.4% of its
parameters in its encoder keeps the count overestimation and most of the
spurious components of the reference architecture, so the improvement is
associated with the pretrained Swin encoder and not with the encoder share alone.

## Layout

```
swinbfe/            the model and the datasets
  net.py            Swin-BFE; Eqs. 1-3
  context_block.py  the state-space context block; Eqs. 5-8
  baselines.py      ResNet-34 U-Net
  rescbam_unet.py   the reference architecture, rebuilt from its public code
  swin_unet.py      an equal-encoder control, not an arm in the paper
scripts/            training, evaluation and analysis
records/            one CSV per run, one row per test image
splits/             the dataset splits and index files
DATASET_VERSION.md  which WHU release was used, with file checksums
```

## Data layout

The index files in `splits/` list every tile with paths relative to one data
root, which is given to each script as `--data-root` (or the environment
variable `SWINBFE_DATA_ROOT`; default: the current directory). The root is
expected to look like this:

```
<data-root>/
  whu_building/
    train/10000.TIF ...                       the official COCO-format release
    converted/train/masks/10000.png ...       binary masks rasterised from the annotations
    converted/train/inst/10000.png ...        uint16 instance maps
    (the same for val/ and test/)
  inria/
    tiles/images/austin1_r0c0.png ...         1024x1024 tiles cut by scripts/prep_inria.py
    tiles/masks/austin1_r0c0.png ...
    tiles/inst/austin1_r0c0.png ...
```

`scripts/prep_whu.py --data-root <data-root>` rasterises the COCO annotations of the
three official archives into `converted/`, giving each annotated building its own
instance identifier, and rewrites `splits/whu_index.json` and `splits/whu_split.json`
(the shipped files are what it produces; the partitions are the dataset's own).
The Inria tiling and split are specific to this study;
`scripts/prep_inria.py --src <inria/AerialImageDataset/train> --dst <data-root>/inria/tiles
--data-root <data-root>` rebuilds the tiles and rewrites `splits/inria_*.json`
with relative paths.

## Install

```
pip install -r requirements.txt
```

`mamba_ssm` is needed only for the two scanning context blocks; the
configuration reported as Swin-BFE runs without it. Its selective-scan kernel
is CUDA-only, so those two arms cannot be evaluated on a CPU.

## Reproduce the paper's numbers without a GPU

Every table and figure rests on the per-image records in `records/`, one row
per test image with true positives, false positives, false negatives, instance
counts and Boundary IoU. Recomputing a published value means reading those
CSVs; no model and no GPU are involved.

`records/` holds 52 files:

| files | what they are |
|---|---|
| `whu_<arm>_a05_s<k>_eval_test_perimage.csv` (15) | five arms times three seeds on WHU |
| `inria_<arm>_a05_s<k>_eval_test_perimage.csv` (12) | four arms times three seeds on Inria |
| `whu_rescbam_refproto2_s<k>_eval_test_perimage.csv` (3) | the reference architecture retrained under its own protocol, scored at the native resolution |
| `whu_rescbam_refproto2_s<k>_eval_test_resize_perimage.csv` (3) | the same models scored at the matched resolution (`--resize-eval`) |
| `whu_<arm>_a05_s<k>_on_inria_perimage.csv` (12) | the WHU-trained checkpoints applied to the Inria test tiles without retraining (cross-domain, Sec. IV-E) |
| `whu_{ctxnone,ours}_a05_s<k>_spurious.csv` (6) | number and total area of the spurious components on each WHU test image (Sec. IV-B) |
| `eval_humg_perimage.csv` (1) | the released reference checkpoint on its own released test set |

```
python scripts/paired_stats.py --base rescbam --arm ctxnone   # Table III, first row
python scripts/macro_variants.py                              # Fig. 9(a), conventions A-E
python scripts/rule_sensitivity.py                            # Sec. IV-C: 44 of 48 verdicts do not depend on the multiplier
python scripts/spurious_area.py --summarize                   # Sec. IV-B: 385 +- 15 against 431 +- 45 px
```

`macro_variants.py` labels the conventions as in the paper: A is the pooled
building-class IoU of the main tables, B the macro IoU over pooled counts, and
C, D and E the per-image macro IoU with an absent class scored 1, with the empty
images dropped, and with an absent class scored 0.

## Retrain

With the official archives unpacked under the data root:

```
export SWINBFE_DATA_ROOT=<data-root>
python scripts/prep_whu.py --data-root $SWINBFE_DATA_ROOT   # masks, instance maps, index
python scripts/train.py --context none    --seed 0   # Swin-BFE
python scripts/train.py --context v2      --seed 0   # Swin-BFE-Mamba
python scripts/train.py --context vanilla --seed 0   # vanilla scan model
python scripts/train.py --model rescbam   --seed 0   # reference architecture
python scripts/train.py --model unet      --seed 0   # ResNet-34 U-Net
```

The defaults of `train.py` are the paper's settings, identical for every arm:
AdamW at learning rate 1e-4 and weight decay 1e-2, batch size 4 with gradient
accumulation 2, 512x512 random crops, 20,000 iterations, bf16 autocast,
ImageNet normalisation, binary cross-entropy plus a Tversky term with
alpha = beta = 0.5. Each arm was trained three times, with seeds 0, 1 and 2.
Inria runs add `--index splits/inria_index.json --split splits/inria_split.json`.
The protocol-transfer check of Sec. IV-F is
`--model rescbam --ref-proto --lr 1e-3 --iters 37000`, evaluated with
`--resize-eval`.

Then evaluate (add `--data-root` or set the environment variable as above):

```
python scripts/eval_whu.py --ckpt runs/<arm>/best.pt --split test
python scripts/eval_whu.py --ckpt runs/<arm>/best.pt --split test --thin
python scripts/verify_config.py <run name>            # checks a checkpoint against the paper
```

The cross-domain records come from the WHU checkpoints scored on the Inria test
tiles, and the spurious-component areas from the WHU predictions (a GPU is
needed for both; `--record` makes the second check its counts against the
shipped record image by image):

```
python scripts/eval_whu.py --ckpt runs/whu_<arm>_a05_s<k>/best.pt --split test \
    --index splits/inria_index.json --splitjson splits/inria_split.json \
    --out records/whu_<arm>_a05_s<k>_on_inria.json
python scripts/spurious_area.py --ckpt runs/whu_<arm>_a05_s<k>/best.pt \
    --record records/whu_<arm>_a05_s<k>_eval_test_perimage.csv \
    --out records/whu_<arm>_a05_s<k>_spurious.csv
```

## Arms

| run tag   | arm in the paper     | flag                | context parameters |
|-----------|----------------------|---------------------|--------------------|
| `ctxnone` | Swin-BFE             | `--context none`    | 0.50 M |
| `ours`    | Swin-BFE-Mamba       | `--context v2`      | 0.28 M |
| `vanilla` | vanilla scan model   | `--context vanilla` | 0.21 M, WHU only |
| `rescbam` | reference architecture | `--model rescbam` | - |
| `unet`    | U-Net (ResNet-34)    | `--model unet`      | - |

`scripts/verify_config.py` checks a checkpoint against the configuration
reported in the paper, including that the upsampling path carries no
parameters.

## A note on checkpoint naming

Checkpoints written during development store `args.model == "cracknet"`, the
name this code had at the time. `"swinbfe"` is the current name and the default
for new runs; both select the same architecture, so checkpoints from either
load without conversion. Likewise the ResNet-34 U-Net checkpoints store
`"unet_rgbd"`, the name of a variant with a depth input that the paper does not
use; `"unet"` is the current name, and both select the same class. Those
checkpoints also carry ten zero-initialised 1x1 projections (`fuse.*`) from
that variant; the class keeps them, unused, so that the checkpoints load with
`strict=True` and the parameter count matches Table I.

## Third-party code

`rescbam_unet.py` and `swin_unet.py` are derived from
<https://github.com/trungdungtdct/Astro_CBAM_Unet> (extracted 2026-08-31) and
carry their own terms. `swin_unet.py` is provided for completeness and is not
one of the arms reported in the paper: the source notebook defines but never
instantiates the class, so its width, block count and patch size were chosen
here.

## License

TODO: choose a license before publishing, and check it against the terms of the
third-party code above.

## Citation

TODO: add the citation once the paper has a DOI.
