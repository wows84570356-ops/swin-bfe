# Which release of the WHU building dataset this code uses

The WHU building dataset is distributed in several releases whose tile size, ground
sampling distance and building counts differ. Numbers obtained on one release are not
comparable with numbers obtained on another, so this file records which one was used and
how to confirm that a local copy holds the same bytes.

## The release

**The aerial subset, COCO-format release at 0.2 m ground sampling distance, 1024 x 1024
tiles**, downloaded from the authors' official page on **2026-08-31**:
<http://gpcv.whu.edu.cn/data/building_dataset.html>, section *Aerial imagery dataset*.
That page offers the aerial subset in three forms:

| Form on the official page | Tile | GSD | |
|---|---|---|---|
| Cropped tiles + raster labels | 512 x 512, 8,189 tiles | 0.3 m | the version most papers use |
| `whu_raster_1024.rar` | 1024 x 1024 | 0.2 m | raster labels |
| `train.zip` / `validation.zip` / `test.zip` / `annotation.zip` | 1024 x 1024 | 0.2 m | **COCO format, used here** |

## Checksums

| File | Bytes | MD5 |
|---|---:|---|
| `train.zip` | 8,249,745,683 | |
| `validation.zip` | 1,652,636,986 | |
| `test.zip` | 6,027,523,194 | |
| `annotation.zip` | 17,979,029 | `c838faa727104aedf4af6de258081c02` |
| `train/10000.TIF` | | `3b95475d7b94b593280bfa5cff7cf85d` |
| `validation/20000.TIF` | | `50816c6cd113cd5597a2d3889e40ed4c` |
| `test/3000.TIF` | | `415081c35727ecda72f4d5f0dfbce887` |

## Contents

| Partition | Tiles | COCO annotations (buildings) |
|---|---:|---:|
| train | 2,943 | 143,252 |
| validation | 627 | 15,926 |
| test | 2,220 | 70,063 |
| **total** | **5,790** | **229,241** |

Every COCO `images` record gives `width: 1024, height: 1024`. Measured from the GeoTIFF
labels, the ground sampling distance is 0.2 m/px (204.8 m per tile) in NZGD2000 / New
Zealand Transverse Mercator 2000, covering Christchurch, New Zealand.

The train, validation and test partitions arrive as three separate archives, so the split
is the dataset's own. A perceptual-hash audit over all 5,790 tiles found no near-duplicate
crossing a partition boundary.

This release must not be compared with the 0.3 m release: that one has 8,189 tiles of
512 x 512 with different building totals. The 1024 x 1024 tiling is also why 24.1% of
the test tiles contain no building, the composition property behind the scoring-convention
analysis in the paper.

## Cross-check

`scripts/prep_whu.py` rasterises each annotated polygon with its own instance identifier.
Summing `n_gt` over any WHU test record in `records/` gives 70,063, the number of test
annotations in the COCO file, so the instance ground truth is the released annotation.
