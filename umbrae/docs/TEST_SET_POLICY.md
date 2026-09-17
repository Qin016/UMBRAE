# Test-set policy — dual-branch protocol v1

The test manifest may be built for integrity auditing, but test metrics are
sealed during Stage A/Stage B development and future LoRA selection.

Test data must not be used for UOT hyperparameter selection, fusion-loss
selection, LoRA rank selection, checkpoint selection, early stopping, or any
other model choice. Evaluation tools default to `--split val`; requesting
`--split test` additionally requires the explicit `--allow-test` flag.

Test evaluation is permitted only for a final, preregistered model or a
predeclared frozen baseline comparison. All methods in a comparison must use
the same test candidate pool and locked protocol version.
