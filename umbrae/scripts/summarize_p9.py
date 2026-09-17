#!/usr/bin/env python
"""Create the locked validation-only P9 scientific summary."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    "UMBRAE": ROOT / "dual_branch_outputs/umbrae_s1_p9_interface_adapter_base_seed42_protocolv1",
    "P6 Full Real": ROOT / "dual_branch_outputs/dualbranch_s1_p9_interface_adapter_p6real_seed42_protocolv1",
}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def selected_interface(run, selected):
    if selected is None:
        return None
    epoch = int(selected["epoch"])
    return next(row for row in run["epochs"] if int(row["epoch"]) == epoch)


def interface_effect(run):
    selected = run["final"]["selected"]
    if selected is None:
        return "NEGATIVE"
    baseline = run["baseline"]
    cider_gain = selected["caption"]["CIDEr"] - baseline["caption"]["CIDEr"]
    grounding_gain = selected["grounding"]["grounding_accuracy"] - baseline["grounding"]["grounding_accuracy"]
    return "POSITIVE" if cider_gain > 0 and grounding_gain > 0 else "WEAK"


def load_run(path):
    final = read_json(path / "final_val_metrics.json")
    run = {
        "path": path,
        "config": read_json(path / "config.json"),
        "provenance": read_json(path / "provenance.json"),
        "epochs": read_jsonl(path / "metrics_per_epoch.jsonl"),
        "downstream": read_jsonl(path / "validation_downstream.jsonl"),
        "baseline_interface": read_json(path / "source_interface_baseline.json"),
        "baseline": read_json(path / "source_downstream_baseline.json"),
        "final": final,
    }
    run["selected_interface"] = selected_interface(run, final["selected"])
    return run


def gains(run):
    baseline = run["baseline"]
    selected = run["final"]["selected"]
    if selected is None:
        return None
    source_gap = run["baseline_interface"]["projected_oracle_gap"]
    gap = selected["projected_oracle_gap"]
    return {
        "projected_gap_recovery_ratio": (source_gap - gap) / source_gap,
        "cider_gain": selected["caption"]["CIDEr"] - baseline["caption"]["CIDEr"],
        "grounding_gain": selected["grounding"]["grounding_accuracy"] - baseline["grounding"]["grounding_accuracy"],
        "cider_oracle_recovery_fraction": (selected["caption"]["CIDEr"] - baseline["caption"]["CIDEr"]) / (1.635679 - baseline["caption"]["CIDEr"]),
        "grounding_oracle_recovery_fraction": (selected["grounding"]["grounding_accuracy"] - baseline["grounding"]["grounding_accuracy"]) / (0.523357 - baseline["grounding"]["grounding_accuracy"]),
    }


def main():
    runs = {name: load_run(path) for name, path in OUTPUTS.items()}
    effects = {name: interface_effect(run) for name, run in runs.items()}
    base_selected = runs["UMBRAE"]["final"]["selected"]
    p6_selected = runs["P6 Full Real"]["final"]["selected"]
    if p6_selected is not None and base_selected is None:
        information_gain = "POSITIVE"
    elif p6_selected is None and base_selected is not None:
        information_gain = "NEGATIVE"
    elif p6_selected is None:
        information_gain = "WEAK"
    elif (p6_selected["projected_oracle_gap"] < base_selected["projected_oracle_gap"] and
          p6_selected["caption"]["CIDEr"] > base_selected["caption"]["CIDEr"] and
          p6_selected["grounding"]["grounding_accuracy"] > base_selected["grounding"]["grounding_accuracy"]):
        information_gain = "POSITIVE"
    elif (p6_selected["projected_oracle_gap"] >= base_selected["projected_oracle_gap"] and
          p6_selected["caption"]["CIDEr"] < base_selected["caption"]["CIDEr"] and
          p6_selected["grounding"]["grounding_accuracy"] < base_selected["grounding"]["grounding_accuracy"]):
        information_gain = "NEGATIVE"
    else:
        information_gain = "WEAK"

    any_selected = base_selected is not None or p6_selected is not None
    both_selected = base_selected is not None and p6_selected is not None
    if not any_selected:
        p9_status = "PROJECTED_DISTANCE_NOT_SUFFICIENT_FOR_DECODER_UTILITY"
        hypothesis = "NOT_SUPPORTED"
    elif information_gain == "POSITIVE":
        p9_status = "P6_INFORMATION_RECOVERABLE_BY_INTERFACE_ADAPTER"
        hypothesis = "SUPPORTED"
    else:
        p9_status = "GENERIC_INTERFACE_ADAPTATION_WORKS"
        hypothesis = "PARTIALLY_SUPPORTED"
    frozen_content = "SUFFICIENT" if information_gain == "POSITIVE" else ("PARTIAL" if any_selected else "INSUFFICIENT")
    simple_adapter = "SUFFICIENT" if any_selected else "INSUFFICIENT"

    summary = {
        "protocol": "protocol_v1 validation only",
        "test_used": False,
        "runs": {},
        "P6_INFORMATION_GAIN": information_gain,
        "INTERFACE_MISMATCH_HYPOTHESIS": hypothesis,
        "FROZEN_P6_INFORMATION_CONTENT": frozen_content,
        "SIMPLE_TOKENWISE_ADAPTER": simple_adapter,
        "P9_STATUS": p9_status,
    }
    for name, run in runs.items():
        summary["runs"][name] = {
            "selection_status": run["final"]["selection_status"],
            "selected": run["final"]["selected"],
            "source_interface": run["baseline_interface"],
            "source_downstream": run["baseline"],
            "adapter_gain": gains(run),
            "interface_adapter_effect": effects[name],
        }

    lines = [
        "# P9 Pre-Projector Interface Adapter Report",
        "",
        "## 1. Setup",
        "",
        "P9 uses train (8,559) and locked validation (300) only. The Brain representation, P6 Structural/Fusion/LoRA components, Shikra `mm_projector`, Shikra decoder, CLIP teacher, tokenizer, prompts, generation, parser, and evaluator remained frozen/unchanged. Only the pre-projector adapter and its scalar residual gate were optimized. No post-hoc test was run or used for selection.",
        "",
        "Two matched runs used seed 42, identical sample order, architecture, initialization, teacher, loss, AdamW settings, batch size 64, and the same 20-epoch cap.",
        "",
        "## 2. Adapter Architecture",
        "",
        "`LayerNorm(1024) → Linear(1024,256) → GELU → Linear(256,1024)` with `Z_hat = Z + sigmoid(gate_logit) * delta`. The final linear was zero initialized and `gate_logit=-2.2`.",
        "",
        "- Adapter parameters excluding gate: 527,616",
        "- Gate parameters: 1",
        "- Total P9-trainable parameters: 527,617",
        "- Trainable / BrainX base: 0.360147%",
        "",
        "## 3. Initialization Equivalence",
        "",
        "| Representation | Max abs error | Mean abs error | Cosine | Initial gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        init = run["provenance"]["initialization_equivalence"]
        lines.append(f"| {name} | {init['max_abs_error']:.8f} | {init['mean_abs_error']:.8f} | {init['cosine_similarity']:.8f} | {init['gate']:.8f} |")

    for number, name in ((4, "UMBRAE"), (5, "P6 Full Real")):
        run = runs[name]
        downstream = {int(row["epoch"]): row for row in run["downstream"] if row["label"] != "source_baseline"}
        lines += ["", f"## {number}. {name} + Adapter Training", "", "| Epoch | Projected gap | Projected cosine | CIDEr | Grounding acc | Gate | Correction ratio |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for epoch in run["epochs"]:
            row = downstream.get(int(epoch["epoch"]))
            lines.append(f"| {epoch['epoch']} | {epoch['projected_oracle_gap']:.6f} | {epoch['projected_paired_token_cosine']:.6f} | {fmt(None if row is None else row['caption']['CIDEr'])} | {fmt(None if row is None else row['grounding']['grounding_accuracy'])} | {epoch['gate']:.6f} | {epoch['adapter_correction_ratio']:.6f} |")

    lines += [
        "",
        "## 6. Interface Recovery",
        "",
        "| Brain representation | Adapter | Projected gap | Gap recovery | Projected cosine | Norm ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        source = run["baseline_interface"]
        lines.append(f"| {name} | None | {source['projected_oracle_gap']:.6f} | 0.000000 | {source['projected_paired_token_cosine']:.6f} | {source['projected_norm_ratio']:.6f} |")
        selected = run["final"]["selected"]
        interface = run["selected_interface"]
        gain = gains(run)
        if selected is None:
            lines.append(f"| {name} | Interface Adapter | no compatible checkpoint | — | — | — |")
        else:
            lines.append(f"| {name} | Interface Adapter (epoch {selected['epoch']}) | {selected['projected_oracle_gap']:.6f} | {gain['projected_gap_recovery_ratio']:.6f} | {interface['projected_paired_token_cosine']:.6f} | {interface['projected_norm_ratio']:.6f} |")
    lines.append("| Exact CLIP Oracle | — | 0 | identity | 1 | 1 |")

    lines += ["", "## 7. Representation Metrics", "", "| Representation | Adapter | Spearman RSA | Pearson RSA | Hard R@1 | R@5 | R@10 | MRR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        for adapter_name, metric in (("None", run["baseline_interface"]), ("Interface Adapter", run["selected_interface"])):
            if metric is None:
                lines.append(f"| {name} | {adapter_name} | — | — | — | — | — | — |")
            else:
                rsa, retrieval = metric["rsa"], metric["hard_retrieval"]
                lines.append(f"| {name} | {adapter_name} | {rsa['spearman_rsa']:.6f} | {rsa['pearson_rsa']:.6f} | {retrieval['recall_at_1']:.6f} | {retrieval['recall_at_5']:.6f} | {retrieval['recall_at_10']:.6f} | {retrieval['mrr']:.6f} |")
    lines += ["", "RSA and hard retrieval are diagnostics only and were not used for checkpoint selection.", "", "## 8. Validation Downstream", "", "| Brain representation | Adapter | CIDEr | BLEU-4 | ROUGE-L | Mean IoU | Grounding acc | Parse failures |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        for adapter_name, row in (("None", run["baseline"]), ("Interface Adapter", run["final"]["selected"])):
            if row is None:
                lines.append(f"| {name} | {adapter_name} | no compatible checkpoint | — | — | — | — | — |")
            else:
                lines.append(f"| {name} | {adapter_name} | {row['caption']['CIDEr']:.6f} | {row['caption']['BLEU-4']:.6f} | {row['caption']['ROUGE-L']:.6f} | {row['grounding']['mean_iou']:.6f} | {row['grounding']['grounding_accuracy']:.6f} | {row['grounding']['parse_failures']} |")
    lines += [
        "| Exact CLIP Oracle | — | not run on validation | — | — | — | — | — |",
        "",
        "Checkpoint compatibility required both `CIDEr >= source CIDEr` and `grounding accuracy >= source grounding accuracy`; among compatible checkpoints, the lowest projected gap was selected. The earlier Exact CLIP values (CIDEr 1.635679, grounding 0.523357) came from reused test and are used only as explicitly labeled recovery-reference constants, not as P9 validation evidence or selection data.",
        "",
        "## 9. Does P6 contain more usable information?",
        "",
        f"`P6_INFORMATION_GAIN = {information_gain}`",
        "",
        "This label compares only downstream-compatible selected checkpoints. A missing compatible checkpoint is not replaced by an interface-only best checkpoint.",
        "",
        "## 10. Interface Adapter Effect",
        "",
        f"`UMBRAE_INTERFACE_ADAPTER = {effects['UMBRAE']}`",
        "",
        f"`P6_INTERFACE_ADAPTER = {effects['P6 Full Real']}`",
        "",
        "## 11. Oracle Headroom Recovery",
        "",
        "| Representation | Gap recovery ratio | CIDEr recovery fraction | Grounding recovery fraction |",
        "|---|---:|---:|---:|",
    ]
    for name, run in runs.items():
        gain = gains(run)
        if gain is None:
            lines.append(f"| {name} | — | — | — |")
        else:
            lines.append(f"| {name} | {gain['projected_gap_recovery_ratio']:.6f} | {gain['cider_oracle_recovery_fraction']:.6f} | {gain['grounding_oracle_recovery_fraction']:.6f} |")
    lines += [
        "",
        "## 12. Scientific Interpretation",
        "",
        f"`INTERFACE_MISMATCH_HYPOTHESIS = {hypothesis}`",
        "",
        f"`FROZEN_P6_INFORMATION_CONTENT = {frozen_content}`",
        "",
        f"`SIMPLE_TOKENWISE_ADAPTER = {simple_adapter}`",
        "",
        f"`P9_STATUS = {p9_status}`",
        "",
        "The result is validation-only. No P9 post-hoc reused-test evaluation was run, no test metric influenced training or selection, and no next-stage token-mixing or joint fine-tuning was started.",
    ]

    report = ROOT / "dual_branch_outputs/p9_pre_projector_interface_adapter_report.md"
    report.write_text("\n".join(lines) + "\n")
    (ROOT / "dual_branch_outputs/p9_pre_projector_interface_adapter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"report": str(report), "P9_STATUS": p9_status, "test_used": False}))


if __name__ == "__main__":
    main()
