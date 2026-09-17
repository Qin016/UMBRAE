# NeuroRoute v1 ROI set

NeuroRoute v1 uses eight subject-specific ROIs aligned to the UMBRAE
`nsdgeneral.npy` vector space:

`V1`, `V2`, `V3`, `hV4`, `FFA`, `EBA`, `PPA`, and `OPA`.

The official NSD `floc-places:RSC` label exists in wholebrain
`func1pt8mm` space, but has zero overlap with the UMBRAE `nsdgeneral`
mask for subjects S1, S2, S5, and S7. Consequently, no real RSC voxel
index can be produced for the current `nsdgeneral.npy` model input.

RSC is therefore recorded as optional but excluded from
NeuroRoute-v1-8ROI. It must not be approximated, borrowed from another
ROI, or represented by fabricated indices. Supporting RSC later requires
an expanded fMRI input space or direct wholebrain input and corresponding
training/data-pipeline changes.

The canonical configuration is
`configs/neuroroute_roi_set_v1_8roi.json`.
