"""torchrun --standalone --nproc_per_node=N tests/mp_check.py <scenario>   (gloo/CPU)"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.distributed as dist
from forgemind import *
from forgemind.distributed import init_distributed
from forgemind.moe import MoE
from forgemind.parallel import native

sc = sys.argv[1]
MC = dict(vocab_size=64, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, max_seq=32)

def train(par, moe=False, steps=30, micro=8, quant_cfg=None, **tc):
    cfg = ForgeConfig(); cfg.parallel.__dict__.update(par)
    cfg.train.__dict__.update(max_steps=steps, warmup_steps=3, lr=3e-3, log_every=1000, bf16=False, resume=False, **tc)
    mc = GPTConfig(**MC, n_experts=4 if moe else 0, top_k=2)
    ds = SyntheticLM(mc.vocab_size, mc.max_seq, n=512)
    tr = Trainer(lambda c: GPT(mc, moe_kwargs(c, cfg) if moe else None), cfg, ds, micro_batch=micro)
    h = tr.fit()
    return tr, h

def report(name, tr, h):
    a, b = h[0]["loss"], h[-1]["loss"]
    if tr.ctx.rank == 0:
        print(f"RESULT {name}: loss {a:.3f} -> {b:.3f}", flush=True)
    assert b < 0.7 * a, (name, a, b)

if sc == "ddp":
    tr, h = train(dict(dp_replicate=2)); report(sc, tr, h)
    for p in tr.raw.parameters():   # replicas must stay identical
        t = p.detach().clone(); ref = t.clone(); dist.broadcast(ref, 0); assert torch.allclose(t, ref, atol=1e-6)
    if tr.ctx.rank == 0: print("RESULT ddp replicas identical")
elif sc == "ddp_accum_offload":
    tr, h = train(dict(dp_replicate=2), grad_accum=2); report(sc, tr, h)
elif sc == "fsdp":
    tr, h = train(dict(dp_shard=2)); report(sc, tr, h)
elif sc == "hsdp":
    tr, h = train(dict(dp_replicate=2, dp_shard=2)); report(sc, tr, h)
elif sc == "fsdp_accum":
    tr, h = train(dict(dp_shard=2), grad_accum=2); report(sc, tr, h)
elif sc == "ep_ddp":
    tr, h = train(dict(dp_replicate=2, ep=2), moe=True); report(sc, tr, h)
elif sc == "ep_ddp4":
    tr, h = train(dict(dp_replicate=4, ep=2), moe=True, steps=30); report(sc, tr, h)
elif sc == "pp":
    tr, h = train(dict(pp=2, pp_microbatches=2), steps=40); report(sc, tr, h)
elif sc == "tp":
    tr, h = train(dict(tp=2)); report(sc, tr, h)
elif sc == "tp_fsdp":
    tr, h = train(dict(tp=2, dp_shard=2)); report(sc, tr, h)
elif sc == "pp_fsdp":
    tr, h = train(dict(pp=2, dp_shard=2, pp_microbatches=2), steps=40); report(sc, tr, h)
elif sc == "3d_full":   # pp x tp x fsdp on 8 ranks
    tr, h = train(dict(pp=2, tp=2, dp_shard=2, pp_microbatches=2), steps=30); report(sc, tr, h)
elif sc == "3d":   # pp x tp x dp(fsdp) needs 8 ranks; here pp=2,tp=2 (+dp=1) on 4
    tr, h = train(dict(pp=2, tp=2, pp_microbatches=2), steps=40); report(sc, tr, h)
elif sc == "ep_equiv":
    ctx = init_distributed(ParallelConfig(dp_replicate=2, ep=2))
    torch.manual_seed(0)
    full = MoE(32, 64, 4, 2)
    loc = MoE(32, 64, 4, 2, ep_size=2, ep_group=ctx.ep_group, ep_rank=ctx.ep_rank)
    loc.router.load_state_dict(full.router.state_dict())
    sl = slice(ctx.ep_rank * 2, ctx.ep_rank * 2 + 2)
    for n in ("w1", "w3", "w2"):
        getattr(loc.experts, n).data.copy_(getattr(full.experts, n).data[sl])
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(2, 6, 32, generator=g) for _ in range(2)]
    out_loc = loc(xs[ctx.rank])
    out_ref = full(xs[ctx.rank])
    assert torch.allclose(out_loc, out_ref, atol=1e-5), (out_loc - out_ref).abs().max()
    (out_loc.pow(2).sum()).backward()
    # reference: sum of both ranks' losses on the full model
    sum(full(x).pow(2).sum() for x in xs).backward()
    for n in ("w1", "w3", "w2"):
        gl, gr = getattr(loc.experts, n).grad, getattr(full.experts, n).grad[sl]
        assert torch.allclose(gl, gr, atol=1e-4), (n, (gl - gr).abs().max())
    if ctx.rank == 0: print("RESULT ep_equiv: forward + expert grads match single-rank reference")
elif sc == "tp_equiv":
    ctx = init_distributed(ParallelConfig(tp=2))
    torch.manual_seed(0)
    mc = GPTConfig(**MC); m = GPT(mc); ref = copy.deepcopy(m)
    native.apply_tp(m, ctx)
    x = torch.randint(0, 64, (2, 16))
    a, b = m(x), ref(x)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    (a.sum()).backward(); b.sum().backward()
    from torch.distributed.tensor import DTensor
    for (n, p), (_, q) in zip(m.named_parameters(), ref.named_parameters()):
        gp = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        assert torch.allclose(gp, q.grad, atol=1e-4), n
    if ctx.rank == 0: print("RESULT tp_equiv: logits + all grads match non-TP reference")
elif sc == "pp_equiv":
    from forgemind.parallel import PipelineRunner
    from forgemind.loss import logits_cross_entropy
    for sched in ("gpipe", "1f1b"):
        ctx = init_distributed(ParallelConfig(pp=2)) if sched == "gpipe" else ctx
        torch.manual_seed(0)
        mc = GPTConfig(**MC); m = GPT(mc); ref = copy.deepcopy(m)
        x = torch.randint(0, 64, (8, 16)); y = torch.randint(0, 64, (8, 16))
        m.split_stage(ctx.pp_rank, 2)
        runner = PipelineRunner(m, ctx, logits_cross_entropy, 4, sched)
        pl = runner.step(x, y)
        ref_loss = logits_cross_entropy(ref(x), y); ref_loss.backward()
        if runner.last:
            assert abs(pl.item() - ref_loss.item()) < 1e-4, (pl.item(), ref_loss.item())
        refp = dict(ref.named_parameters())
        for n, p in m.named_parameters():
            assert torch.allclose(p.grad, refp[n].grad, atol=1e-4), (sched, n, (p.grad - refp[n].grad).abs().max())
        if ctx.rank == 0: print(f"RESULT pp_equiv[{sched}]: loss + stage grads match single-process reference", flush=True)
elif sc == "prefetch":
    # Rank skew: every rank must see exactly the batches the serial path would give it,
    # in the same order and count -- the prefetch thread must not reorder or drop anything.
    from forgemind.data import BatchStream
    ctx = init_distributed(ParallelConfig(dp_replicate=2))
    ds = SyntheticLM(64, 32, n=512)
    dl, sampler = build_loader(ds, 8, ctx)
    serial = BatchStream(dl, sampler, ctx.device, threaded=False)
    ref = [serial.next()[0].clone() for _ in range(12)]
    serial.close()
    dl, sampler = build_loader(ds, 8, ctx)
    fast = BatchStream(dl, sampler, ctx.device, threaded=True, depth=3)
    got = [fast.next()[0].clone() for _ in range(12)]
    rep = fast.report()
    fast.close()
    assert len(got) == 12 and all(torch.equal(a, b) for a, b in zip(ref, got)), ctx.rank
    n = torch.tensor([len(got)], dtype=torch.long)
    lens = [torch.zeros_like(n) for _ in range(ctx.world)]
    dist.all_gather(lens, n)
    assert all(int(x) == 12 for x in lens), lens          # no rank lagged or over-ran
    assert rep["threaded"] is True
    mine = ref[0][0, :4].tolist()
    other = [None for _ in range(ctx.world)]
    dist.all_gather_object(other, mine)
    if ctx.rank == 0:
        print(f"RESULT prefetch: 12/12 batches identical to serial per rank, "
              f"wait {rep['wait_ms_per_batch']} ms/batch, stalls {rep['stalled']}", flush=True)
else:
    raise SystemExit("unknown scenario")
dist.barrier(); cleanup()
