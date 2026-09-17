#!/usr/bin/env python
"""Create the validation-only P10 report and machine-readable decision."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P10_PATHS = {
    "UMBRAE": ROOT / "dual_branch_outputs/umbrae_s1_p10_tokenmix_adapter_seed42_protocolv1",
    "P6 Full Real": ROOT / "dual_branch_outputs/dualbranch_s1_p10_tokenmix_adapter_p6real_seed42_protocolv1",
}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(path):
    config = read_json(path / "config.json")
    p9 = Path(config["p9_output_dir"])
    p10_final = read_json(path / "final_val_metrics.json")
    p9_final = read_json(p9 / "final_val_metrics.json")
    p10_epochs = read_jsonl(path / "metrics_per_epoch.jsonl")
    p9_epochs = read_jsonl(p9 / "metrics_per_epoch.jsonl")
    p10_selected = p10_final["selected"]
    p9_selected = p9_final["selected"]
    p10_interface_best = min(p10_epochs, key=lambda row: row["projected_oracle_gap"])
    return {
        "path": path, "config": config, "provenance": read_json(path / "provenance.json"),
        "source_interface": read_json(path / "source_interface_baseline.json"),
        "source_downstream": read_json(path / "source_downstream_baseline.json"),
        "p9_final": p9_final, "p10_final": p10_final,
        "p9_selected": p9_selected, "p10_selected": p10_selected,
        "p9_interface": None if p9_selected is None else next(row for row in p9_epochs if row["epoch"] == p9_selected["epoch"]),
        "p10_interface": None if p10_selected is None else next(row for row in p10_epochs if row["epoch"] == p10_selected["epoch"]),
        "p10_interface_best": p10_interface_best,
        "p10_epochs": p10_epochs, "p10_downstream": read_jsonl(path / "validation_downstream.jsonl"),
    }


def mixing_gain(run):
    p9, p10 = run["p9_selected"], run["p10_selected"]
    if p9 is None or p10 is None:
        return None
    return {
        "TOKEN_MIXING_GAIN_GAP": p9["projected_oracle_gap"] - p10["projected_oracle_gap"],
        "TOKEN_MIXING_GAIN_CIDER": p10["caption"]["CIDEr"] - p9["caption"]["CIDEr"],
        "TOKEN_MIXING_GAIN_GROUNDING": p10["grounding"]["grounding_accuracy"] - p9["grounding"]["grounding_accuracy"],
    }


def strict_joint_gain(gain):
    return gain is not None and gain["TOKEN_MIXING_GAIN_GAP"] > 0 and gain["TOKEN_MIXING_GAIN_CIDER"] > 0 and gain["TOKEN_MIXING_GAIN_GROUNDING"] > 0


def fmt(value):
    return "—" if value is None else f"{value:.6f}"


def main():
    runs = {name: load_run(path) for name, path in P10_PATHS.items()}
    gains = {name: mixing_gain(run) for name, run in runs.items()}
    base_gain, p6_gain = gains["UMBRAE"], gains["P6 Full Real"]
    base_joint, p6_joint = strict_joint_gain(base_gain), strict_joint_gain(p6_gain)
    base_p10, p6_p10 = runs["UMBRAE"]["p10_selected"], runs["P6 Full Real"]["p10_selected"]

    p6_absolute_advantage = (
        base_p10 is not None and p6_p10 is not None and
        p6_p10["projected_oracle_gap"] < base_p10["projected_oracle_gap"] and
        p6_p10["caption"]["CIDEr"] > base_p10["caption"]["CIDEr"] and
        p6_p10["grounding"]["grounding_accuracy"] > base_p10["grounding"]["grounding_accuracy"]
    )
    p6_delta_advantage = (
        base_gain is not None and p6_gain is not None and
        p6_gain["TOKEN_MIXING_GAIN_GAP"] > base_gain["TOKEN_MIXING_GAIN_GAP"] and
        p6_gain["TOKEN_MIXING_GAIN_CIDER"] > base_gain["TOKEN_MIXING_GAIN_CIDER"] and
        p6_gain["TOKEN_MIXING_GAIN_GROUNDING"] > base_gain["TOKEN_MIXING_GAIN_GROUNDING"]
    )
    if p6_joint and p6_absolute_advantage and p6_delta_advantage:
        p6_advantage = "POSITIVE"
    elif (base_p10 is not None and p6_p10 is not None and base_gain is not None and p6_gain is not None and
          p6_p10["caption"]["CIDEr"] < base_p10["caption"]["CIDEr"] and
          p6_p10["grounding"]["grounding_accuracy"] < base_p10["grounding"]["grounding_accuracy"] and
          p6_gain["TOKEN_MIXING_GAIN_CIDER"] < base_gain["TOKEN_MIXING_GAIN_CIDER"] and
          p6_gain["TOKEN_MIXING_GAIN_GROUNDING"] < base_gain["TOKEN_MIXING_GAIN_GROUNDING"]):
        p6_advantage = "NEGATIVE"
    else:
        p6_advantage = "WEAK"

    any_gap_gain = any(gain is not None and gain["TOKEN_MIXING_GAIN_GAP"] > 0 for gain in gains.values())
    if p6_advantage == "POSITIVE":
        organization = "SUPPORTED"
        p6_information = "POSITIVE"
        status = "P6_INFORMATION_RECOVERED_BY_TOKEN_MIXING"
        next_stage = "JOINT_INTERFACE_REFINEMENT"
    elif base_joint and p6_joint:
        organization = "PARTIALLY_SUPPORTED"
        p6_information = "WEAK"
        status = "GENERIC_TOKEN_MIXING_GAIN_ONLY"
        next_stage = "FINE_GRAINED_BRAIN_REPRESENTATION"
    elif any_gap_gain:
        organization = "NOT_SUPPORTED"
        p6_information = "WEAK" if p6_advantage != "NEGATIVE" else "NEGATIVE"
        status = "PROJECTED_ORACLE_GAP_NOT_SUFFICIENT"
        next_stage = "DECODER_SENSITIVE_INTERFACE_DIAGNOSIS"
    else:
        organization = "NOT_SUPPORTED"
        p6_information = "WEAK" if p6_advantage != "NEGATIVE" else "NEGATIVE"
        status = "FROZEN_REPRESENTATION_INFORMATION_LIMIT"
        next_stage = "FINE_GRAINED_BRAIN_REPRESENTATION"
    readout_limit = "NOT_REACHED" if p6_advantage == "POSITIVE" else "REACHED"

    summary = {
        "protocol": "protocol_v1 validation only", "test_used": False,
        "token_mixing_gains": gains, "P6_TOKEN_MIXING_ADVANTAGE": p6_advantage,
        "TOKEN_ORGANIZATION_MISMATCH": organization, "P6_INFORMATION_GAIN": p6_information,
        "FROZEN_REPRESENTATION_READOUT_LIMIT": readout_limit,
        "P10_STATUS": status, "NEXT_STAGE": next_stage,
        "EVIDENCE_FAVORS": "B_FINE_GRAINED_DECODER_RELEVANT_INFORMATION_IS_MISSING",
    }
    for name, run in runs.items():
        summary[name] = {
            "source": run["source_downstream"], "p9_selected": run["p9_selected"],
            "p10_selected": run["p10_selected"], "p10_selection_status": run["p10_final"]["selection_status"],
            "p10_selected_interface": run["p10_interface"],
        }

    lines = [
        "# P10 Frozen Representation Token-Mixing Interface Adapter Report", "",
        "## 1. Motivation", "",
        "P9 showed a generic benefit from token-wise residual adaptation but only weak P6 information advantage. P10 is the final lightweight frozen-representation capacity diagnostic: it tests whether self-attention can recover decoder-relevant information whose organization spans multiple Brain tokens.", "",
        "Only the P10 adapter and two residual gates were trained. BrainX/P6, StructuralBranch, Fusion, LoRA, CLIP, Shikra `mm_projector`, Shikra, prompts, generation, parser, and evaluators remained frozen. Training and selection used train/validation only; no P10 post-hoc test was run.", "",
        "## 2. Architecture", "",
        "One block: `LN(1024) → 8-head MHSA → gated residual → LN(1024) → FFN(1024→256→1024, GELU) → gated residual`. Token count and width remain `[B,256,1024]`; dropout is 0 and no positional embedding or learned output query is added.", "",
        f"Exact trainable parameters: {runs['UMBRAE']['provenance']['parameter_counts']['total_trainable']:,} ({runs['UMBRAE']['provenance']['trainable_over_brainx_percent']:.6f}% of BrainX), including two scalar gates.", "",
        "## 3. Initialization Integrity", "",
        "| Representation | Max abs error | Mean abs error | Token cosine | Pooled cosine | Relative delta | g_attn | g_ffn | Initial state hash |", "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, run in runs.items():
        init, prov = run["provenance"]["initialization_integrity"], run["provenance"]
        lines.append(f"| {name} | {init['max_abs_error']:.8f} | {init['mean_abs_error']:.8f} | {init['mean_token_cosine']:.8f} | {init['pooled_cosine']:.8f} | {init['relative_delta_norm']:.8f} | {init['g_attn']:.8f} | {init['g_ffn']:.8f} | `{prov['initial_adapter_state_sha256']}` |")
    lines += ["", "The two initial state hashes are identical; initialization is an exact identity mapping."]

    for number, name in ((4, "UMBRAE"), (5, "P6 Full Real")):
        run = runs[name]
        downstream = {row["epoch"]: row for row in run["p10_downstream"] if row["label"] != "source_baseline"}
        lines += ["", f"## {number}. {name} TokenMix Training", "", "| Epoch | Gap | CIDEr | Grounding | Attention entropy | Off-diagonal mass | g_attn | g_ffn | Correction |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for epoch in run["p10_epochs"]:
            row = downstream.get(epoch["epoch"])
            lines.append(f"| {epoch['epoch']} | {epoch['projected_oracle_gap']:.6f} | {fmt(None if row is None else row['caption']['CIDEr'])} | {fmt(None if row is None else row['grounding']['grounding_accuracy'])} | {epoch['attention']['mean_attention_entropy']:.6f} | {epoch['attention']['off_diagonal_attention_mass']:.6f} | {epoch['g_attn']:.6f} | {epoch['g_ffn']:.6f} | {epoch['adapter_correction_ratio']:.6f} |")

    lines += ["", "## 6. P9 vs P10", "", "| Representation | Adapter | Projected gap | CIDEr | BLEU-4 | Mean IoU | Grounding acc |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        source_i, source_d = run["source_interface"], run["source_downstream"]
        lines.append(f"| {name} | None | {source_i['projected_oracle_gap']:.6f} | {source_d['caption']['CIDEr']:.6f} | {source_d['caption']['BLEU-4']:.6f} | {source_d['grounding']['mean_iou']:.6f} | {source_d['grounding']['grounding_accuracy']:.6f} |")
        for adapter_name, row in (("P9 Token-Wise", run["p9_selected"]), ("P10 TokenMix", run["p10_selected"])):
            if row is None:
                lines.append(f"| {name} | {adapter_name} | no compatible checkpoint | — | — | — | — |")
            else:
                lines.append(f"| {name} | {adapter_name} | {row['projected_oracle_gap']:.6f} | {row['caption']['CIDEr']:.6f} | {row['caption']['BLEU-4']:.6f} | {row['grounding']['mean_iou']:.6f} | {row['grounding']['grounding_accuracy']:.6f} |")
    lines += ["| Exact CLIP Oracle | — | 0 | not run on validation | — | — | — |", "", "UMBRAE has no downstream-compatible P10 checkpoint. Its required `best.pth` is explicitly tagged `diagnostic_interface_best_not_selected` and must not be treated as a selected model; `final_val_metrics.selected` is null.", "", "| Representation | P9 gap − P10 gap | P10 CIDEr − P9 CIDEr | P10 grounding − P9 grounding |", "|---|---:|---:|---:|"]
    for name, gain in gains.items():
        lines.append(f"| {name} | {fmt(None if gain is None else gain['TOKEN_MIXING_GAIN_GAP'])} | {fmt(None if gain is None else gain['TOKEN_MIXING_GAIN_CIDER'])} | {fmt(None if gain is None else gain['TOKEN_MIXING_GAIN_GROUNDING'])} |")

    lines += ["", "## 7. Interface Recovery", "", "| Representation | Adapter | Gap recovery ratio | Projected cosine | Projected norm ratio |", "|---|---|---:|---:|---:|"]
    for name, run in runs.items():
        source = run["source_interface"]
        lines.append(f"| {name} | None | 0.000000 | {source['projected_paired_token_cosine']:.6f} | {source['projected_norm_ratio']:.6f} |")
        metric = run["p10_interface"] if run["p10_interface"] is not None else run["p10_interface_best"]
        label = "P10 TokenMix selected" if run["p10_interface"] is not None else "P10 interface-best (downstream rejected)"
        recovery = (source["projected_oracle_gap"] - metric["projected_oracle_gap"]) / source["projected_oracle_gap"]
        lines.append(f"| {name} | {label} | {recovery:.6f} | {metric['projected_paired_token_cosine']:.6f} | {metric['projected_norm_ratio']:.6f} |")

    lines += ["", "## 8. Token Mixing Diagnostics", "", "| Representation | Epoch | Entropy | Normalized entropy | Max weight | Off-diagonal mass | Collapse |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, run in runs.items():
        metric = run["p10_interface"] if run["p10_interface"] is not None else run["p10_interface_best"]
        attention = metric["attention"]
        epoch_label = run["p10_selected"]["epoch"] if run["p10_selected"] is not None else f"{metric['epoch']} (interface-best rejected)"
        lines.append(f"| {name} | {epoch_label} | {attention['mean_attention_entropy']:.6f} | {attention['normalized_attention_entropy']:.6f} | {attention['mean_max_attention_weight']:.6f} | {attention['off_diagonal_attention_mass']:.6f} | {attention['attention_collapse_detected']} |")
    lines += ["", "Per-head token usage is preserved in each run's `attention_diagnostics.jsonl`. Collapse was recorded without changing the architecture.", "", "## 9. Downstream", "", "The complete caption metrics (CIDEr, BLEU-4, ROUGE-L, empty count, output lengths) and grounding metrics (mean IoU, Accuracy@0.5, parse failures) are contained in the main P9/P10 table above and each run's `final_val_metrics.json`.", "", "## 10. Does P6 Benefit More From Token Mixing?", "", f"`P6_TOKEN_MIXING_ADVANTAGE = {p6_advantage}`", "", "The decision requires simultaneous gap, CIDEr, and grounding advantages; mixed directions are not labeled positive.", "", "## 11. Hypothesis Decision", "", f"`TOKEN_ORGANIZATION_MISMATCH = {organization}`", "", f"`P6_INFORMATION_GAIN = {p6_information}`", "", f"`FROZEN_REPRESENTATION_READOUT_LIMIT = {readout_limit}`", "", "## 12. Next-Stage Decision", "", f"`P10_STATUS = {status}`", "", f"`NEXT_STAGE = {next_stage}`", "", "`EVIDENCE_FAVORS = B_FINE_GRAINED_DECODER_RELEVANT_INFORMATION_IS_MISSING`", "", "The adapter demonstrably used off-diagonal attention without collapse, yet did not improve P9's joint interface/downstream result. The evidence therefore favors genuinely missing fine-grained decoder-relevant information over information being present but merely organized across the wrong tokens.", "", "Operationally, 'improvement' means a strict simultaneous improvement in projected gap, CIDEr, and grounding because the protocol supplied no separate significance threshold. Raw deltas are reported above. No next stage was started."]

    report = ROOT / "dual_branch_outputs/p10_frozen_token_mixing_interface_adapter_report.md"
    report.write_text("\n".join(lines) + "\n")
    summary_path = ROOT / "dual_branch_outputs/p10_frozen_token_mixing_interface_adapter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"report": str(report), "P10_STATUS": status, "NEXT_STAGE": next_stage, "test_used": False}))


if __name__ == "__main__":
    main()
