"""Bounded CPU regressions: real native transformers, synthetic images/text/VAE."""
import argparse
import ast
import contextlib
import json
import inspect
import io
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator

from flux2.controlnet_flux2klein import FLUX2KleinControlNetModel
from flux2.pipeline_flux2klein_controlnet import Flux2KleinControlNetPipeline
from flux2.transformer_flux2klein import Flux2Transformer2DModel

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "train_flux2klein_controlnet.py"
INFER_SOURCE = ROOT / "infer_flux2klein_controlnet.py"
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float32)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]


def process_memory_bytes():
    if sys.platform == "linux":
        import resource

        # Linux reports ru_maxrss in KiB, not Windows private-commit bytes.
        return dict(peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if sys.platform != "win32":
        raise RuntimeError("The test memory guard supports Windows and Linux only")
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(peak_working_set=counters.PeakWorkingSetSize,
                peak_private_commit=counters.PeakPagefileUsage, private=counters.PrivateUsage)


def cpu_resources():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized():
        raise RuntimeError("Tests require CUDA_VISIBLE_DEVICES=-1 and uninitialized CUDA")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Tests require Hugging Face and Transformers offline controls")
    memory = process_memory_bytes()
    if max(memory.values()) > 2 * 2**30:
        raise MemoryError("Conservative 2 GiB process memory guard exceeded")
    return dict({name + "_MiB": round(value / 2**20, 2) for name, value in memory.items()},
                cuda_initialized=False)


def setUpModule():
    cpu_resources()


def tearDownModule():
    print(json.dumps(dict(cpu_resources=cpu_resources())), flush=True)


class MemorySafeguards(unittest.TestCase):
    def test_linux_peak_rss_units(self):
        resource = types.SimpleNamespace(RUSAGE_SELF=0,
                                        getrusage=mock.Mock(return_value=types.SimpleNamespace(ru_maxrss=1536 * 1024)))
        with mock.patch.object(sys, "platform", "linux"), mock.patch.dict(sys.modules, resource=resource):
            self.assertEqual(process_memory_bytes(), dict(peak_rss=1536 * 2**20))
            self.assertEqual(cpu_resources()["peak_rss_MiB"], 1536)
        resource.getrusage.assert_called_with(resource.RUSAGE_SELF)

    def test_two_gib_boundary_for_each_memory_counter(self):
        for name in ("peak_rss", "peak_working_set", "peak_private_commit", "private"):
            with self.subTest(counter=name):
                with mock.patch(__name__ + ".process_memory_bytes", return_value={name: 2 * 2**30}):
                    self.assertEqual(cpu_resources()[name + "_MiB"], 2048)
                with mock.patch(__name__ + ".process_memory_bytes", return_value={name: 2 * 2**30 + 1}):
                    with self.assertRaisesRegex(MemoryError, "2 GiB"):
                        cpu_resources()

    def test_unsupported_platform_does_not_skip_memory_guard(self):
        with mock.patch.object(sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Windows and Linux"):
                process_memory_bytes()


def definition(path, name):
    # Avoid executing training imports/entry points (including unavailable Prodigy).
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    namespace = dict(globals())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def assignment(path, function_name, target):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    return next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in n.targets))


def execute(nodes, namespace, path=TRAIN_SOURCE):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class FixtureVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(block_out_channels=(8, 8, 8, 8), batch_norm_eps=1e-4)
        self.bn = torch.nn.BatchNorm2d(128, affine=False).eval()
        self.inputs = []

    @property
    def dtype(self):
        return self.bn.running_mean.dtype

    def encode(self, image):
        self.inputs.append(image.detach().clone())
        latent = F.avg_pool2d(image.mean(dim=1, keepdim=True), 8).expand(-1, 32, -1, -1).clone()
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: latent))


def models(double_blocks=4, single_blocks=24):
    base = Flux2Transformer2DModel(
        in_channels=128, out_channels=128, num_layers=double_blocks, num_single_layers=single_blocks,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=48,
        timestep_guidance_channels=32, mlp_ratio=1.0, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False,
    )
    base.requires_grad_(False)
    control = FLUX2KleinControlNetModel.from_transformer(
        base, num_layers=2, num_single_layers=10, attention_head_dim=16, num_attention_heads=2,
    )
    return base, control


def pipeline(base, control):
    return Flux2KleinControlNetPipeline(
        scheduler=None, vae=FixtureVAE(), text_encoder=None, tokenizer=None, transformer=base, controlnet=control,
    )


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.random.default_generator.manual_seed(20261010)
        self.base, self.control = models()
        self.pipe = pipeline(self.base, self.control)
        self.kwargs = dict(batch_size=1, generator=None, device=torch.device("cpu"), dtype=torch.float32)

    def test_condition_width_and_mask_channels(self):
        image = torch.linspace(-1, 1, 64 * 64).reshape(1, 1, 64, 64).repeat(1, 3, 1, 1)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :, :32] = 1
        one, _ = self.pipe.prepare_control_latents([image], [mask], **self.kwargs)
        three, _ = self.pipe.prepare_control_latents([image], [mask.repeat(1, 3, 1, 1)], **self.kwargs)
        self.assertEqual(tuple(one.shape), (1, 16, 131))
        self.assertTrue(torch.equal(one, three))
        self.assertEqual(tuple(self.control.controlnet_x_embedder.weight.shape), (32, 131))
        self.assertEqual(tuple(self.control.controlnet_x_embedder(one).shape), (1, 16, 32))
        legacy_129 = torch.cat([one[..., :128], one[..., 128:129]], dim=-1)
        with self.assertRaises(RuntimeError):
            self.control.controlnet_x_embedder(legacy_129)
        self.assertTrue(torch.equal(one[..., 128:], one[..., 128:129].expand(-1, -1, 3)))
        with self.assertRaises(ValueError):
            self.pipe.prepare_control_latents([image], [mask.repeat(1, 2, 1, 1)], **self.kwargs)

    def test_training_inference_preprocessing(self):
        rgb = np.stack([np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))] * 3, axis=-1)
        rgb[:, -1, :] = 255
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, :32] = 255
        source_pil, mask_pil = Image.fromarray(rgb), Image.fromarray(mask)
        with tempfile.TemporaryDirectory(prefix="phase3b_images_") as temp_name:
            root = Path(temp_name).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            self.assertNotIn(ROOT, root.parents)
            source_pil.save(root / "source.png")
            mask_pil.save(root / "mask.png")
            item = dict(source_images=str(root / "source.png"), control_images=str(root / "mask.png"),
                        target_images=str(root / "source.png"), prompts="change [TARGET]")
            (root / "data.json").write_text(json.dumps([item]), encoding="utf-8")
            dataset = definition(TRAIN_SOURCE, "ControlNetImageEditDataset")(str(root / "data.json"), base_resolution=4096)
            sample = dataset[0]
            source = sample["source_images"].unsqueeze(0)
            target = sample["target_images"].unsqueeze(0)
            train_mask = sample["control_images"].unsqueeze(0)
            infer_source = self.pipe.prepare_images([source_pil], self.pipe.image_processor)[0]
            infer_mask = self.pipe.prepare_images([mask_pil], self.pipe.mask_processor)[0]
            self.assertTrue(torch.equal(source, infer_source))
            self.assertTrue(torch.equal(source, target))
            self.assertEqual((float(source.min()), float(source.max())), (-1.0, 1.0))
            self.assertEqual(tuple(train_mask.shape), (1, 3, 64, 64))
            self.assertEqual(set(train_mask.unique().tolist()), {0.0, 1.0})
            self.assertTrue(torch.equal(train_mask, infer_mask.repeat(1, 3, 1, 1)))
            train_condition, _ = self.pipe.prepare_control_latents([source], [train_mask], **self.kwargs)
            train_vae_input = self.pipe.vae.inputs[-1]
            infer_condition, _ = self.pipe.prepare_control_latents([infer_source], [infer_mask], **self.kwargs)
            self.assertTrue(torch.equal(train_condition, infer_condition))
            self.assertTrue(torch.equal(train_vae_input, self.pipe.vae.inputs[-1]))
            self.assertTrue(torch.equal(train_vae_input[:, :, :, :32], source[:, :, :, :32]))
            self.assertTrue(torch.equal(train_vae_input[:, :, :, 32:], -torch.ones_like(source[:, :, :, 32:])))
            displayed = np.array(T.ToPILImage()((sample["source_images"] + 1) / 2)).astype(np.int16)
            self.assertLessEqual(int(np.abs(displayed - rgb.astype(np.int16)).max()), 1)
        self.assertFalse(root.exists())

    def test_mask_threshold(self):
        source = torch.full((1, 3, 64, 64), 0.25)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :, :16] = 0.499
        mask[:, :, :, 16:32] = 0.5
        mask[:, :, :, 32:48] = 1
        condition, _ = self.pipe.prepare_control_latents([source], [mask], **self.kwargs)
        encoded = self.pipe.vae.inputs[-1]
        self.assertTrue(torch.equal(encoded[:, :, :, :16], -torch.ones_like(encoded[:, :, :, :16])))
        self.assertTrue(torch.equal(encoded[:, :, :, 16:48], source[:, :, :, 16:48]))
        self.assertGreaterEqual(float(encoded.min()), -1)
        self.assertLessEqual(float(encoded.max()), 1)
        self.assertEqual(set(condition[..., 128:].unique().tolist()), {0.0, 1.0})

    def test_target_source_positional_roles(self):
        image = torch.zeros(1, 3, 64, 64)
        noise, noise_ids = self.pipe.prepare_latents(
            batch_size=1, num_latents_channels=32, height=64, width=64, dtype=torch.float32,
            device=torch.device("cpu"), generator=None,
        )
        source, source_ids = self.pipe.prepare_image_latents([image], **self.kwargs)
        target, _ = self.pipe.prepare_image_latents([image], **self.kwargs)
        namespace = dict(torch=torch, noise_ids=noise_ids, source_ids=source_ids)
        # Execute the actual trainer ID assignment, rather than reproduce its intended expression.
        execute([assignment(TRAIN_SOURCE, "train", "latent_image_ids")], namespace)
        ids = namespace["latent_image_ids"]
        self.assertEqual(tuple(noise.shape), (1, 16, 128))
        self.assertEqual(noise.shape, source.shape)
        self.assertEqual(noise.shape, target.shape)
        self.assertEqual(tuple(ids.shape), (1, 32, 4))
        self.assertTrue(torch.equal(ids[0, :16, 0], torch.zeros(16)))
        self.assertTrue(torch.equal(ids[0, 16:, 0], torch.full((16,), 10.0)))
        expected_spatial = torch.tensor([[row, col, 0] for row in range(4) for col in range(4)])
        self.assertTrue(torch.equal(ids[0, :16, 1:], expected_spatial))
        self.assertTrue(torch.equal(ids[0, :16, 1:], ids[0, 16:, 1:]))
        self.assertFalse(torch.equal(ids[:, :16], ids[:, 16:]))


class SigmaSelection(unittest.TestCase):
    def setUp(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0,
                                                         use_dynamic_shifting=False)
        self.get_sigmas = definition(TRAIN_SOURCE, "get_sigmas")

    def test_bf16_collisions_do_not_change_index_selection(self):
        rounded = self.scheduler.timesteps.to(torch.bfloat16)
        self.assertEqual(rounded.unique().numel(), 506)
        for timestep in (999.0, 1000.0):
            # Original equality lookup: distinct scheduler entries collapse before nonzero().item().
            with self.assertRaises(RuntimeError):
                (rounded == torch.tensor(timestep, dtype=torch.bfloat16)).nonzero().item()
        indices = torch.arange(1000, dtype=torch.long)
        actual = self.get_sigmas(self.scheduler, indices, 3, torch.device("cpu"), torch.bfloat16)
        self.assertEqual(tuple(actual.shape), (1000, 1, 1))
        self.assertTrue(torch.equal(actual.flatten(), self.scheduler.sigmas[:1000].to(torch.bfloat16)))
        high = self.get_sigmas(self.scheduler, torch.tensor([0, 1]), 3, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.equal(self.scheduler.timesteps[:2], torch.tensor([1000.0, 999.0])))
        self.assertTrue(torch.equal(high.flatten(), self.scheduler.sigmas[:2]))
        self.assertAlmostEqual(high[0].item(), 1.0)
        self.assertAlmostEqual(high[1].item(), 0.999, places=7)
        self.assertEqual(assignment(TRAIN_SOURCE, "train", "sigmas").value.args[1].id, "indices")

    def test_scheduler_bounds_and_index_type(self):
        for indices in (torch.tensor([-1]), torch.tensor([1000])):
            with self.assertRaisesRegex(IndexError, "outside"):
                self.get_sigmas(self.scheduler, indices, 3, torch.device("cpu"), torch.bfloat16)
        for indices in (torch.tensor([1.0]), torch.tensor([[1]])):
            with self.assertRaisesRegex(ValueError, "integer"):
                self.get_sigmas(self.scheduler, indices, 3, torch.device("cpu"), torch.bfloat16)
        self.scheduler.sigmas = self.scheduler.sigmas[:1]
        with self.assertRaisesRegex(ValueError, "fewer sigmas"):
            self.get_sigmas(self.scheduler, torch.tensor([0]), 3, torch.device("cpu"), torch.bfloat16)


class TransformerGradients(unittest.TestCase):
    def setUp(self):
        torch.random.default_generator.manual_seed(20261010)
        self.base, self.control = models()
        self.pipe = pipeline(self.base, self.control)
        grid, source_grid = torch.randn(1, 128, 2, 2), torch.randn(1, 128, 2, 2)
        clean = self.pipe._pack_latents(grid)
        noise = torch.randn(1, 4, 128)
        self.target = noise - clean
        hidden = torch.cat([0.6 * clean + 0.4 * noise, self.pipe._pack_latents(source_grid)], dim=1)
        text = torch.randn(1, 4, 48)
        self.condition = torch.randn(1, 4, 131)
        self.common = dict(
            hidden_states=hidden, timestep=torch.tensor([0.4]), encoder_hidden_states=text,
            img_ids=torch.cat([self.pipe._prepare_latent_ids(grid), self.pipe._prepare_image_ids([source_grid])], dim=1),
            txt_ids=self.pipe._prepare_text_ids(text), guidance=None, return_dict=False,
        )

    def predict(self, control=None, condition=None):
        control = self.control if control is None else control
        condition = self.condition if condition is None else condition
        double, single = control(controlnet_cond=condition, conditioning_scale=1.0, **self.common)
        return self.base(controlnet_block_samples=double, controlnet_single_block_samples=single,
                         **self.common)[0][:, :4]

    def group_norm(self, prefix):
        params = [p for name, p in self.control.named_parameters() if name.startswith(prefix)]
        self.assertTrue(params)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
        return math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))

    def test_zero_baseline_staged_gradients_and_condition_sensitivity(self):
        frozen = {name: p.detach().clone() for name, p in self.base.named_parameters()}
        baseline = self.base(**self.common)[0][:, :4].detach()
        initial = self.predict().detach()
        self.assertTrue(torch.equal(baseline, initial))
        self.assertTrue(torch.equal(initial, self.predict(condition=self.condition + 0.75).detach()))
        optimizer = torch.optim.SGD(self.control.parameters(), lr=0.2)
        stages = []
        for stage in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (self.predict() - self.target).square().mean()
            loss.backward()
            double_norms = [float(head.weight.grad.double().norm()) for head in self.control.controlnet_blocks]
            single_norms = [float(head.weight.grad.double().norm()) for head in self.control.controlnet_single_blocks]
            self.assertTrue(all(value > 0 and math.isfinite(value) for value in double_norms + single_norms))
            upstream = self.group_norm("transformer_blocks.")
            late_blocks = [self.group_norm("single_transformer_blocks." + str(i) + ".") for i in (8, 9)]
            condition_norm = self.group_norm("controlnet_x_embedder.")
            if stage == 0:
                self.assertEqual(upstream, 0)
                self.assertEqual(late_blocks, [0, 0])
                self.assertEqual(condition_norm, 0)
            else:
                self.assertGreater(upstream, 0)
                self.assertTrue(all(value > 0 for value in late_blocks))
                self.assertGreater(condition_norm, 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.control.parameters() if p.grad is not None))
            self.assertTrue(all(p.grad is None for p in self.base.parameters()))
            stages.append(dict(stage=stage, loss=float(loss.detach()), double_heads=double_norms,
                               single_heads=single_norms, upstream_double=upstream,
                               upstream_single_8_9=late_blocks, conditioning=condition_norm))
            if stage < 2:
                optimizer.step()
        sensitivity = float((self.predict(condition=self.condition + 0.75) - self.predict()).detach().abs().max())
        self.assertGreater(sensitivity, 1e-7)
        self.assertTrue(all(torch.equal(p, frozen[name]) for name, p in self.base.named_parameters()))
        self.assertTrue(all(p.device.type == "cpu" for model in (self.base, self.control) for p in model.parameters()))
        print(json.dumps(dict(gradient_stages=stages, condition_sensitivity_max_abs=sensitivity,
                              zero_baseline_max_abs=float((initial - baseline).abs().max()))), flush=True)

    def test_ordinary_and_checkpointed_backward(self):
        # Open the zero heads with one controlled update before comparing upstream gradients.
        optimizer = torch.optim.SGD(self.control.parameters(), lr=0.2)
        (self.predict() - self.target).square().mean().backward()
        optimizer.step()
        self.control.zero_grad(set_to_none=True)
        ordinary = self.predict()
        (ordinary - self.target).square().mean().backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone()
                     for name, p in self.control.named_parameters()}
        self.control.zero_grad(set_to_none=True)
        self.base.enable_gradient_checkpointing()
        self.control.enable_gradient_checkpointing()
        checked = self.predict()
        (checked - self.target).square().mean().backward()
        torch.testing.assert_close(ordinary, checked, atol=1e-6, rtol=1e-5)
        max_gradient_error = 0.0
        for name, param in self.control.named_parameters():
            self.assertEqual(gradients[name] is None, param.grad is None, name)
            if param.grad is not None:
                torch.testing.assert_close(gradients[name], param.grad, atol=1e-6, rtol=1e-5)
                max_gradient_error = max(max_gradient_error, float((gradients[name] - param.grad).abs().max()))
        print(json.dumps(dict(checkpoint_output_max_abs=float((ordinary - checked).detach().abs().max()),
                              checkpoint_gradient_max_abs=max_gradient_error)), flush=True)

    def check_routing(self, base, stream, head_count):
        blocks = base.single_transformer_blocks if stream == "single" else base.transformer_blocks
        pre, post, terminal, handles = {}, {}, {}, []
        for index, block in enumerate(blocks):
            def before(module, args, kwargs, index=index):
                pre[index] = kwargs["hidden_states"].detach().clone()
            def after(module, args, output, index=index):
                post[index] = (output if stream == "single" else output[1]).detach().clone()
            handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(block.register_forward_hook(after))
        if stream == "single":
            handles.append(base.norm_out.register_forward_pre_hook(
                lambda module, args: terminal.update(value=args[0].detach().clone())))
        else:
            handles.append(base.single_transformer_blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: terminal.update(value=kwargs["hidden_states"][:, 4:].detach().clone()),
                with_kwargs=True))
        sentinels = [torch.full((1, 8, 32), 0.01 * (index + 1)) for index in range(head_count)]
        name = "controlnet_single_block_samples" if stream == "single" else "controlnet_block_samples"
        try:
            base(**{name: sentinels}, **self.common)
        finally:
            for handle in handles:
                handle.remove()
        mapping = []
        max_error = 0.0
        for index in range(len(blocks)):
            next_value = pre[index + 1] if index + 1 < len(blocks) else terminal["value"]
            previous = post[index]
            if stream == "single":
                next_value = next_value[:, 4:] if index + 1 < len(blocks) else next_value
                previous = previous[:, 4:]
            head = index * head_count // len(blocks)
            self.assertGreaterEqual(head, 0)
            self.assertLess(head, head_count)
            error = float((next_value - previous - 0.01 * (head + 1)).abs().max())
            self.assertLessEqual(error, 1e-5)
            max_error = max(max_error, error)
            mapping.append(head)
        self.assertEqual(set(mapping), set(range(head_count)))
        print(json.dumps(dict(stream=stream, mapping=mapping, routing_max_abs_error=max_error)), flush=True)
        return mapping

    def test_single_stream_24_to_10_reachability(self):
        mapping = self.check_routing(self.base, "single", 10)
        self.assertEqual([mapping.count(i) for i in range(10)], [3, 2, 3, 2, 2, 3, 2, 3, 2, 2])

    def test_double_stream_8_to_4_mapping_unchanged(self):
        base, _ = models(double_blocks=8)
        mapping = self.check_routing(base, "double", 4)
        self.assertEqual(mapping, [index // math.ceil(8 / 4) for index in range(8)])


class CheckpointIntegrity(unittest.TestCase):
    def setUp(self):
        torch.random.default_generator.manual_seed(20261010)
        self.base, self.control = models()
        self.base.to(torch.bfloat16)
        self.control.to(torch.bfloat16)
        self.pipe = pipeline(self.base, self.control)
        for head in list(self.control.controlnet_blocks) + list(self.control.controlnet_single_blocks):
            torch.nn.init.normal_(head.weight, std=0.01)
        torch.nn.init.normal_(self.control.controlnet_x_embedder.weight, std=0.01)
        self.optimizer = torch.optim.SGD(self.control.parameters(), lr=0.001, momentum=0.9)
        param = self.control.controlnet_blocks[0].weight
        param.grad = torch.ones_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        grid = torch.zeros(1, 128, 2, 2)
        text = torch.randn(1, 4, 48).to(torch.bfloat16)
        self.condition = torch.randn(1, 4, 131).to(torch.bfloat16)
        self.common = dict(hidden_states=torch.randn(1, 8, 128).to(torch.bfloat16),
                           timestep=torch.tensor([0.4], dtype=torch.bfloat16), encoder_hidden_states=text,
                           img_ids=torch.cat([self.pipe._prepare_latent_ids(grid),
                                              self.pipe._prepare_image_ids([grid])], dim=1),
                           txt_ids=self.pipe._prepare_text_ids(text), guidance=None, return_dict=False)

    def predict(self, control):
        double, single = control(controlnet_cond=self.condition, **self.common)
        return self.base(controlnet_block_samples=double, controlnet_single_block_samples=single,
                         **self.common)[0].detach()

    def checkpoint_block(self, final):
        tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"))
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        label = "lastest" if final else "checkpoint-"
        block = next(n for n in ast.walk(train) if isinstance(n, ast.If) and n.body
                     and isinstance(n.body[0], ast.Assign) and label in ast.unparse(n.body[0]))
        save_index = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Expr)
                          and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "save_pretrained")
        return block.body[:save_index + 1]

    def check_roundtrip(self, final):
        before = self.predict(self.control)
        states = {name: value.detach().clone() for name, value in self.control.state_dict().items()}
        placements = [(id(p), p.dtype, p.device) for p in self.control.parameters()]
        param = self.control.controlnet_blocks[0].weight
        momentum = self.optimizer.state[param]["momentum_buffer"]
        momentum_id, momentum_before = id(momentum), momentum.clone()
        groups = [[id(p) for p in group["params"]] for group in self.optimizer.param_groups]
        with tempfile.TemporaryDirectory(prefix="phase3b_checkpoint_") as temp_name:
            root = Path(temp_name).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            self.assertNotIn(ROOT, root.parents)
            namespace = dict(globals(), checkpoint_dir=str(root / "absent" / "checkpoints"),
                             global_steps=3, align_bit=4, controlnet=self.control,
                             accelerator=types.SimpleNamespace(unwrap_model=lambda model: model),
                             unwrap_model=definition(TRAIN_SOURCE, "unwrap_model"), logger=mock.Mock())
            execute(self.checkpoint_block(final), namespace)
            saved = Path(namespace["save_path"])
            self.assertEqual(saved.name, "lastest" if final else "checkpoint-0003")
            self.assertEqual(sorted(p.name for p in saved.iterdir()),
                             ["config.json", "diffusion_pytorch_model.safetensors"])
            restored = FLUX2KleinControlNetModel.from_pretrained(
                str(saved), local_files_only=True, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16,
            )
            for key in inspect.signature(FLUX2KleinControlNetModel.__init__).parameters:
                if key != "self":
                    self.assertEqual(json.dumps(self.control.config[key]), json.dumps(restored.config[key]), key)
            self.assertEqual(set(states), set(restored.state_dict()))
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(states[name], value), name)
                self.assertEqual(states[name].dtype, value.dtype, name)
            after = self.predict(restored)
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(placements, [(id(p), p.dtype, p.device) for p in self.control.parameters()])
            self.assertTrue(all(torch.equal(states[name], value) for name, value in self.control.state_dict().items()))
            self.assertEqual(groups, [[id(p) for p in group["params"]] for group in self.optimizer.param_groups])
            self.assertEqual(momentum_id, id(self.optimizer.state[param]["momentum_buffer"]))
            self.assertTrue(torch.equal(momentum_before, self.optimizer.state[param]["momentum_buffer"]))
            self.assertEqual(momentum.dtype, torch.bfloat16)
            # Execute the actual inference load assignment without executing its GPU/model-loading entry point.
            infer_tree = ast.parse(INFER_SOURCE.read_text(encoding="utf-8"))
            infer_load = next(n for n in ast.walk(infer_tree) if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "controlnet" for t in n.targets))
            infer_namespace = dict(globals(), args=types.SimpleNamespace(controlnet=str(saved)),
                                   weight_dtype=torch.bfloat16)
            execute([infer_load], infer_namespace, INFER_SOURCE)
            self.assertTrue(torch.equal(before, self.predict(infer_namespace["controlnet"])))
            print(json.dumps(dict(checkpoint="final" if final else "periodic", state_exact=True,
                                  output_max_abs=float((before - after).abs().max()), dtype="bfloat16",
                                  parameter_identity_preserved=True, optimizer_momentum_preserved=True)), flush=True)
        self.assertFalse(root.exists())

    def test_periodic_roundtrip_and_live_optimizer_preservation(self):
        self.check_roundtrip(final=False)

    def test_final_roundtrip_and_live_optimizer_preservation(self):
        self.check_roundtrip(final=True)

    def test_validation_explicit_path_and_invalid_requests(self):
        with tempfile.TemporaryDirectory(prefix="phase3b_validation_") as temp_name:
            root = Path(temp_name).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            self.assertNotIn(ROOT, root.parents)
            saved = root / "supplied"
            self.control.save_pretrained(str(saved))
            tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "log_validation")
            branch = next(n for n in function.body if isinstance(n, ast.If))
            namespace = dict(globals(), args=types.SimpleNamespace(base_model="synthetic", output_dir="unused"),
                             accelerator=mock.Mock(), transformer=self.base, controlnet=self.control,
                             is_final_validation=True, final_checkpoint=str(saved), device=torch.device("cpu"),
                             weight_dtype=torch.bfloat16, Flux2KleinControlNetPipeline=mock.Mock())
            execute([branch], namespace)
            self.assertTrue(torch.equal(self.predict(self.control), self.predict(namespace["controlnet"])))
            namespace["final_checkpoint"] = str(root / "missing")
            with self.assertRaises(OSError):
                execute([branch], namespace)
            infer_tree = ast.parse(INFER_SOURCE.read_text(encoding="utf-8"))
            load = next(n for n in ast.walk(infer_tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "controlnet" for t in n.targets))
            with self.assertRaises(OSError):
                execute([load], dict(globals(), args=types.SimpleNamespace(controlnet=str(root / "missing")),
                                     weight_dtype=torch.bfloat16), INFER_SOURCE)
        self.assertFalse(root.exists())

    def test_inference_requires_checkpoint_argument(self):
        parse_args = definition(INFER_SOURCE, "parse_args")
        with mock.patch.object(sys, "argv", ["infer"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)
        with mock.patch.object(sys, "argv", ["infer", "--controlnet", "supplied"]):
            self.assertEqual(parse_args().controlnet, "supplied")


if __name__ == "__main__":
    unittest.main(verbosity=2)
