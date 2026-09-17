#!/usr/bin/env python
"""Cache fixed P7 test representations and verify the unified encoder."""

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import braceexpand
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.downstream_brain_encoder import MODES, DownstreamBrainEncoder
from models.dual_branch_cache import sha256_file
from models.shared_lora import SharedLoRAUMBRAEEncoder, load_lora_state_dict
from models.umbrae_backbone import FrozenUMBRAEEncoder
from scripts.train_p6_full_stage_c import build_models


def resolve_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("brainx_checkpoint", "p5_checkpoint", "p6_real_checkpoint", "p6_random_checkpoint", "real_structural_checkpoint", "random_structural_checkpoint", "real_roi_mapping", "random_roi_mapping", "validation_fmri", "mm_projector", "shikra_model", "caption_references", "grounding_annotations", "grounding_categories", "output_dir"):
        config[key] = str(Path(config[key]).resolve())
    return config


def load_test_fmri(config, output):
    path = output / "test_fmri_mean_valid_repeats.npy"
    if path.exists():
        values = np.load(path, mmap_mode="r")
        if values.shape != (982, 15724): raise ValueError("Existing P7 fMRI cache shape mismatch")
        return values
    urls = list(braceexpand.braceexpand(config["test_data"])); dataset = wds.WebDataset(urls, resampled=False, cache_dir=str(Path(urls[0]).parent), nodesplitter=lambda x: x).decode("torch").rename(voxels="nsdgeneral.npy").to_tuple("voxels").batched(1, partial=False)
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(982, 15724)); count = 0
    for (voxel,) in torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=1, shuffle=False):
        if count >= 982: raise ValueError("Test webdataset contains too many samples")
        target[count] = voxel.float().mean(1)[0].numpy(); count += 1
    if count != 982: raise ValueError(f"Test webdataset contains {count}, expected 982")
    target.flush(); return np.load(path, mmap_mode="r")


def direct_model(mode, config):
    if mode == "umbrae": return FrozenUMBRAEEncoder(config["brainx_checkpoint"], "subj01")
    if mode == "lora":
        model = SharedLoRAUMBRAEEncoder(FrozenUMBRAEEncoder(config["brainx_checkpoint"], "subj01"), 8, 16, .05); load_lora_state_dict(model, torch.load(config["p5_checkpoint"], map_location="cpu")["lora"]); return model
    kind = mode[5:]
    # P6 checkpoint supplies both final LoRA and Fusion; structural remains P3-R.
    base = FrozenUMBRAEEncoder(config["brainx_checkpoint"], "subj01"); semantic = SharedLoRAUMBRAEEncoder(base, 8, 16, .05); state = torch.load(config[f"p6_{kind}_checkpoint"], map_location="cpu"); load_lora_state_dict(semantic, state["lora"])
    mapping_args = SimpleNamespace(mapping_kind=kind, roi_mapping=config[f"{kind}_roi_mapping"]); from scripts.train_stage_a_structural import active_mapping; _, mapping = active_mapping(mapping_args); from scripts.train_p3r_roi_relational import P3RModel; structural = P3RModel(mapping); structural.load_state_dict(torch.load(config[f"{kind}_structural_checkpoint"], map_location="cpu")["model"]); from models.gated_cross_attention import GatedCrossAttentionFusion; fusion = GatedCrossAttentionFusion(1024,8,0,-4); fusion.load_state_dict(state["fusion"])
    class Direct(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.semantic=semantic; self.structural=structural; self.fusion=fusion
        def forward(self, x):
            z=self.semantic(x); h=self.structural.structural(x)["h_struct"]; return self.fusion(z,h,return_attention=False)["z_cal"]
    return Direct().eval()


def equivalence(config, device):
    fmri = np.load(config["validation_fmri"], mmap_mode="r"); x = torch.from_numpy(np.array(fmri[:2], dtype=np.float32, copy=True)).to(device); results = {}
    for mode in MODES:
        unified = DownstreamBrainEncoder(mode, config).to(device).eval(); direct = direct_model(mode, config).to(device).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")): left, right = unified(x).float(), direct(x).float()
        difference = (left-right).abs(); results[mode] = {"shape": list(left.shape), "all_finite": bool(torch.isfinite(left).all()), "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()), "cosine_similarity": float(F.cosine_similarity(left.flatten(1), right.flatten(1)).mean())}
        del unified, direct; gc.collect(); torch.cuda.empty_cache()
    return results


def load_projector(path, device):
    projector = torch.nn.Linear(1024,4096); state=torch.load(path,map_location="cpu"); projector.load_state_dict({key.split(".")[-1]: value for key,value in state.items()}); return projector.to(device).eval().requires_grad_(False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--device",default="cuda");parser.add_argument("--batch-size",type=int,default=64);args=parser.parse_args(); config=resolve_config(args.config); output=Path(config["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    checks=equivalence(config,args.device); (output/"pre_evaluation_equivalence.json").write_text(json.dumps(checks,indent=2)); print(json.dumps({"equivalence":checks}),flush=True)
    fmri=load_test_fmri(config,output); paths={}
    for mode in MODES:
        path=output/f"{mode}_z_out_fp16.npy"; target=np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(982,256,1024)); model=DownstreamBrainEncoder(mode,config).to(args.device).eval()
        for start in range(0,982,args.batch_size):
            end=min(start+args.batch_size,982); x=torch.from_numpy(np.array(fmri[start:end],dtype=np.float32,copy=True)).to(args.device)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=args.device.startswith("cuda")): target[start:end]=model(x).cpu().numpy().astype(np.float16)
        target.flush();paths[mode]=str(path.resolve());print(json.dumps({"mode":mode,"cached":982}),flush=True);del model,target;gc.collect();torch.cuda.empty_cache()
    projector=load_projector(config["mm_projector"],args.device); baseline=np.load(paths["umbrae"],mmap_mode="r"); diagnostics={}
    for mode in MODES:
        values=np.load(paths[mode],mmap_mode="r"); norm_sum=norm_sq=cos_sum=0.0;tokens=0
        for start in range(0,982,8):
            end=min(start+8,982); z=torch.from_numpy(np.array(values[start:end],dtype=np.float32,copy=True)).to(args.device); b=torch.from_numpy(np.array(baseline[start:end],dtype=np.float32,copy=True)).to(args.device)
            with torch.inference_mode(): projected=projector(z); projected_base=projector(b); norms=projected.float().norm(dim=-1); cosine=F.cosine_similarity(projected.float(),projected_base.float(),dim=-1)
            norm_sum+=float(norms.sum());norm_sq+=float(norms.square().sum());cos_sum+=float(cosine.sum());tokens+=norms.numel()
        mean=norm_sum/tokens; diagnostics[mode]={"projected_shape":[982,256,4096],"projected_token_norm_mean":mean,"projected_token_norm_std":max(norm_sq/tokens-mean*mean,0)**.5,"token_cosine_to_umbrae":cos_sum/tokens}
    metadata={"protocol_version":"protocol_v1","split":"test","sample_count":982,"test_evaluation_started_after_model_selection":True,"representations":paths,"equivalence":checks,"projected_token_diagnostics":diagnostics,"brainx_sha256":sha256_file(config["brainx_checkpoint"]),"mm_projector_sha256":sha256_file(config["mm_projector"])};(output/"feature_cache_metadata.json").write_text(json.dumps(metadata,indent=2));print(json.dumps({"diagnostics":diagnostics}),flush=True)


if __name__=="__main__":main()
