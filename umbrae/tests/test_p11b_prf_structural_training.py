import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from losses.prf_local_alignment_loss import PRFLocalAlignmentLoss
from scripts.train_p11b_prf_structural import score
from umbrae.models.prf_structural_encoder import PRFStructuralEncoder


ROOT=Path(__file__).resolve().parents[1]
B0=ROOT/'dual_branch_outputs/p11b0_prf_grounded_unit_construction'


def units(name):
    p=json.loads((B0/name).read_text());return [u for u in p['units'] if u['unit_type']=='retinotopic']


def param_hash(model):
    d=hashlib.sha256()
    for n,p in model.named_parameters():d.update(n.encode());d.update(str(tuple(p.shape)).encode());d.update(p.detach().numpy().tobytes())
    return d.hexdigest()


def test_output_shape_and_shared_projection_architecture():
    synthetic=[{'unit_id':f'u{i}','parent_roi':'V1','hemisphere':'left','voxel_indices':[2*i,2*i+1]} for i in range(64)]
    model=PRFStructuralEncoder(synthetic,hidden_dim=8,output_dim=16)
    assert model(torch.randn(3,128)).shape==(3,64,16)
    assert len(model.input_projections)==64
    assert not hasattr(model,'unit_embedding')


def test_real_random_capacity_and_initialization_match():
    real=units('selected_mapping_real.json');random=units('selected_mapping_random_seed42.json')
    assert len(real)==len(random)==64
    assert [u['num_voxels'] for u in real]==[u['num_voxels'] for u in random]
    torch.manual_seed(42);a=PRFStructuralEncoder(real)
    torch.manual_seed(42);b=PRFStructuralEncoder(random)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert param_hash(a)==param_hash(b)


def test_teacher_is_identical_and_only_membership_differs():
    real=units('selected_mapping_real.json');random=units('selected_mapping_random_seed42.json')
    assert all(np.array_equal(a['patch_affinity'],b['patch_affinity']) for a,b in zip(real,random))
    assert [u['parent_roi'] for u in real]==[u['parent_roi'] for u in random]
    assert [u['hemisphere'] for u in real]==[u['hemisphere'] for u in random]
    assert any(a['voxel_indices']!=b['voxel_indices'] for a,b in zip(real,random))


def test_all_four_losses_are_finite_and_teacher_has_no_gradient():
    prediction=torch.randn(4,64,32,requires_grad=True);teacher=torch.randn(4,64,32)
    result=PRFLocalAlignmentLoss()(prediction,teacher)
    assert all(torch.isfinite(result[k]) for k in ('loss','local_cos_loss','local_mse','unit_rel_loss','local_nce'))
    result['loss'].backward();assert prediction.grad is not None and teacher.grad is None


def test_checkpoint_selection_is_deterministic_lexicographic():
    a={'unit_localization':{'mrr':.2},'local_sample_retrieval':{'mrr':.1},'local_cos_loss':.5}
    b={'unit_localization':{'mrr':.2},'local_sample_retrieval':{'mrr':.11},'local_cos_loss':.9}
    assert score(b)>score(a) and score(a)==score(a)


def test_training_source_has_no_forbidden_model_import_or_test_loader():
    import scripts.train_p11b_prf_structural as module
    source=inspect.getsource(module)
    assert 'import model' not in source and 'BrainX' not in source and 'Shikra' not in source
    assert "P11BDataset(a.test" not in source and "'test'" not in source


def test_frozen_teacher_cache_declares_no_test_and_correct_shape_if_available():
    path=ROOT/'protocol_outputs/protocol_v1/subj01/p11b_local_teacher_k64/metadata.json'
    if path.exists():
        meta=json.loads(path.read_text());assert meta['test_loaded'] is False
        assert set(meta['splits'])=={'train','val'}
        assert meta['splits']['val']['shape']==[300,64,1024]


def test_completed_runs_emit_64_per_unit_rows_if_available():
    for name in ('p11b_prf_real_k64_seed42_protocolv1','p11b_prf_random_k64_seed42_protocolv1'):
        path=ROOT/'dual_branch_outputs'/name/'per_unit_metrics.csv'
        if path.exists(): assert len(path.read_text().splitlines())==65
