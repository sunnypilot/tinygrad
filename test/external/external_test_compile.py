import tempfile, unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from examples.openpilot.load_pickle import make_inputs
from tinygrad.helpers import Context, DEV, fetch, getenv
from tinygrad.engine.realize import lower_and_compile
from tinygrad.uop.ops import Ops
from examples.openpilot.helpers import load_pickle, make_retargetable


class TestCompile(unittest.TestCase):
  def test_metadata(self):
    import onnx
    from examples.openpilot.compile_onnx import onnx_metadata
    graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['input'], ['output'])], 'metadata',
      [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, (1, 3))],
      [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (1, 3))])
    model = onnx.helper.make_model(graph)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'metadata.onnx'
      for metadata in ({}, {'model_checkpoint': 'v1.0', 'output_slices': 'dGVzdA=='}):
        onnx.helper.set_model_props(model, metadata)
        onnx.save(model, path)
        properties, output_shapes = onnx_metadata(path)
        self.assertEqual(properties, metadata)
        self.assertEqual(output_shapes, {'output': (1, 3)})

  def test_warp_layouts(self):
    from tinygrad import Tensor
    from examples.openpilot.compile_warp import NV12Frame, make_luma_warp, make_frame_prepare
    frame = NV12Frame(8, 6, 12, 8, 4, 144)
    data = np.zeros((12, 12), dtype=np.uint8)
    data[:6, :8] = np.arange(48, dtype=np.uint8).reshape(6, 8)
    data[8:, 0:8:2], data[8:, 1:8:2] = 100, 200
    image = Tensor(data.reshape(-1)).realize()
    transform = np.eye(3, dtype=np.float32)
    luma = make_luma_warp(frame, 4, 4, border_fill=16)
    np.testing.assert_array_equal(luma(input_frame=image, M_inv=Tensor(transform, device='NPY')).numpy(), data[:4, :4].reshape(1, 16))
    transform[0, 2] = 1000
    np.testing.assert_array_equal(luma(input_frame=image, M_inv=Tensor(transform, device='NPY')).numpy(), np.full((1, 16), 16, dtype=np.uint8))
    yuv = make_frame_prepare(frame, 4, 4)
    expected = [[[0, 2], [16, 18]], [[8, 10], [24, 26]], [[1, 3], [17, 19]], [[9, 11], [25, 27]], [[100]*2]*2, [[200]*2]*2]
    np.testing.assert_array_equal(yuv(input_frame=image, M_inv=Tensor(np.eye(3, dtype=np.float32), device='NPY')).numpy(), expected)

  def test_reproject_tables(self):
    from tinygrad import Tensor
    from examples.openpilot.compile_warp import NV12Frame
    from examples.openpilot.compile_reproject_tables import make_tables, nv12_body
    frame = NV12Frame(12, 12, 16, 16, 8, 448)
    fisheye, pinhole = {"f": 8.0, "cx": 6.0, "cy": 6.0, "k": [-0.015, 0.024, -0.009], "tc": 1.5}, {"f": 20.0, "cx": 6.0, "cy": 6.0, "k1": -0.03}
    lenses = {"os_wide": fisheye, "os_narrow": pinhole, "ar_ox_wide": fisheye, "ar_ox_narrow": pinhole}
    tables = make_tables(frame, frame, lenses, zmin=-1.0, feather=4.0, alpha_shift=22, invalid_bit=1 << 30)
    i = np.arange(nv12_body(frame))
    chroma = i >= (uv := frame.stride * frame.y_height)
    row, col = np.where(chroma, i - uv, i) // frame.stride, np.where(chroma, i - uv, i) % frame.stride
    px, plane, size = np.where(chroma, col // 2, col), np.where(chroma, col % 2, 0), np.where(chroma, 6, 12)
    inside = (px < size) & (row < size)
    def index(r, c): return np.where(chroma, uv + r * frame.stride + 2 * c + plane, r * frame.stride + c)
    def split(pw): return pw & ((1 << 22) - 1), (pw >> 22) & 0xff, pw >= 1 << 30

    # the same lenses on both sides: every byte samples itself, and the inset fades in over the feather from the frame edge
    eye = Tensor(np.eye(3, dtype=np.float32))
    pw_wide = tables(eye, "wide")[0].numpy()
    pw_narrow, pn = (t.numpy() for t in tables(eye, "narrow"))
    idx, _, invalid = split(pw_wide)
    np.testing.assert_array_equal(idx[inside], i[inside])
    np.testing.assert_array_equal(invalid, ~inside)
    np.testing.assert_array_equal(pn[inside], i[inside])
    edge = np.minimum.reduce([px + 0.5, size - px - 0.5, row + 0.5, size - row - 0.5]) * np.where(chroma, 2, 1)
    np.testing.assert_array_equal(split(pw_narrow)[1], np.where(inside, np.round(np.clip(edge / 4, 0, 1) * 255), 0))

    # a quarter turn about the optical axis: the wide samples its frame turned with it
    idx, _, invalid = split(tables(Tensor(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)), "wide")[0].numpy())
    np.testing.assert_array_equal(idx[inside], index(px, size - 1 - row)[inside])
    np.testing.assert_array_equal(invalid, ~inside)

    # a larger 3X frame in another layout, its lenses 4 px off center: samples off its frame are invalid and clamped into it
    src, ssize = NV12Frame(16, 16, 20, 18, 10, 600), np.where(chroma, 8, 16)
    suv = src.stride * src.y_height
    def src_index(r, c): return np.where(chroma, suv + r * src.stride + 2 * c + plane, r * src.stride + c)
    self.assertEqual((nv12_body(frame), nv12_body(src)), (384, 560))
    for shift in (-4, 4):
      off = {"ar_ox_wide": fisheye | {"cx": 8.0 + shift, "cy": 8.0 + shift}, "ar_ox_narrow": pinhole | {"cx": 8.0 + shift, "cy": 8.0 + shift}}
      shifted = make_tables(src, frame, lenses | off, zmin=-1.0, feather=4.0, alpha_shift=22, invalid_bit=1 << 30)
      sx, sy = px + (2 + shift) // np.where(chroma, 2, 1), row + (2 + shift) // np.where(chroma, 2, 1)
      ok = inside & (sx >= 0) & (sx < ssize) & (sy >= 0) & (sy < ssize)
      idx, _, invalid = split(shifted(eye, "wide")[0].numpy())
      pw_narrow, pn = (t.numpy() for t in shifted(eye, "narrow"))
      np.testing.assert_array_equal(idx[ok], src_index(sy, sx)[ok])
      np.testing.assert_array_equal(invalid, ~ok)
      np.testing.assert_array_equal(pn[ok], src_index(sy, sx)[ok])
      np.testing.assert_array_equal(split(pw_narrow)[1] > 0, ok)
      np.testing.assert_array_equal(split(pw_narrow)[2], ~inside)
      for t in (idx, pn, split(pw_narrow)[0]):
        np.testing.assert_array_equal((t >= suv) & (t < nv12_body(src)), chroma)
        np.testing.assert_array_less(np.where(chroma, t - suv, t) % src.stride, src.width)

  def test_reproject_tables_real_lenses(self):
    from tinygrad import Tensor
    from examples.openpilot.compile_warp import NV12Frame
    from examples.openpilot.compile_reproject_tables import make_tables
    src, dst = NV12Frame(1928, 1208, 2048, 1216, 608, 4804608), NV12Frame(1344, 760, 1408, 768, 384, 2428928)
    lenses = {"os_wide": {"f": 447.502, "cx": 672.0, "cy": 380.0, "k": [-0.014942, -0.0023814, -0.00064424], "tc": 1.548},
              "os_narrow": {"f": 1156.5, "cx": 672.0, "cy": 380.0, "k1": -0.03},
              "ar_ox_wide": {"f": 597.732, "cx": 963.936, "cy": 603.959, "k": [-0.011968, 0.024043, -0.0091132], "tc": 1.51354},
              "ar_ox_narrow": {"f": 2600.85, "cx": 964.0, "cy": 604.0, "k1": -0.364}}
    v = np.array([0.019, -0.0069, -0.0032])  # a real unit's narrow->wide rotation vector
    K, a = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]), np.linalg.norm(v)
    R = np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a**2 * K @ K
    zmin, feather = -0.0349, 50.0
    tables = make_tables(src, dst, lenses, zmin=zmin, feather=feather, alpha_shift=22, invalid_bit=1 << 30)

    # every comma 4 luma pixel against numpy, which runs the lens models only forwards (ray -> pixel) and inverts them by interpolation
    def theta_d(th, L):
      k, tc = L["k"], L["tc"]
      def p(t): return t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6)
      return np.where(th <= tc, p(th), p(tc) + (1 + 3 * k[0] * tc**2 + 5 * k[1] * tc**4 + 7 * k[2] * tc**6) * (th - tc))
    def project(x, y, z, L):
      rho = np.hypot(x, y)
      r = L["f"] * theta_d(np.arctan2(rho, z), L) / rho if "k" in L else L["f"] / z * (1 + L["k1"] * rho**2 / z**2)
      return L["cx"] + r * x, L["cy"] + r * y
    def unproject(px, py, L):
      dx, dy = (px - L["cx"]) / L["f"], (py - L["cy"]) / L["f"]
      rd, g = np.hypot(dx, dy), np.linspace(0, np.pi if "k" in L else 2, 1 << 20)
      if "k" in L:
        th = np.interp(rd, theta_d(g, L), g)
        return np.sin(th) * dx / rd, np.sin(th) * dy / rd, np.cos(th)
      ru = np.interp(rd, g * (1 + L["k1"] * g * g), g)
      return ru * dx / rd / np.sqrt(1 + ru * ru), ru * dy / rd / np.sqrt(1 + ru * ru), 1 / np.sqrt(1 + ru * ru)
    def sample(mx, my):  # the nearest 3X pixel, whether it's in the frame, and whether it's within a pixel of the frame edge
      xi, yi = np.round(mx - 0.5), np.round(my - 0.5)
      edge = np.minimum.reduce([np.abs(mx), np.abs(mx - src.width), np.abs(my), np.abs(my - src.height)]) < 1
      return xi, yi, (xi >= 0) & (xi < src.width) & (yi >= 0) & (yi < src.height), edge
    def luma(t): return t.numpy()[:dst.stride * dst.height].reshape(dst.height, dst.stride)[:, :dst.width]
    def assert_close(a, b, ok):  # off by one only where float32 lands on the other side of a rounding boundary
      diff = np.abs(a - b)[ok]
      self.assertLessEqual(diff.max(), 1)
      self.assertGreater((diff == 0).mean(), 0.999)

    py, px = np.mgrid[:dst.height, :dst.width] + 0.5
    for cam in ("wide", "narrow"):
      x, y, z = unproject(px, py, lenses[f"os_{cam}"])
      xw, yw, zw = (R[r, 0] * x + R[r, 1] * y + R[r, 2] * z for r in range(3))
      with np.errstate(divide="ignore", invalid="ignore"):
        wx, wy, w_in, w_edge = sample(*project(xw, yw, zw, lenses["ar_ox_wide"]))
      w_ok = w_in & (zw >= zmin)
      pw, pn = (luma(t) for t in tables(Tensor(R.astype(np.float32)), cam))
      idx, alpha, invalid = pw & ((1 << 22) - 1), (pw >> 22) & 0xff, pw >= 1 << 30
      assert_close(idx % src.stride, wx, w_ok)
      assert_close(idx // src.stride, wy, w_ok)
      np.testing.assert_array_equal(invalid[~w_edge & (np.abs(zw - zmin) > 1e-4)], ~w_ok[~w_edge & (np.abs(zw - zmin) > 1e-4)])
      if cam == "wide": continue  # only the narrow output blends in the narrow
      with np.errstate(divide="ignore", invalid="ignore"):
        nx, ny = project(x, y, z, lenses["ar_ox_narrow"])
        nxi, nyi, n_in, n_edge = sample(nx, ny)
      n_ok = n_in & (z > 0)
      assert_close(pn % src.stride, nxi, n_ok)
      assert_close(pn // src.stride, nyi, n_ok)
      # the narrow fades out over `feather` comma 4 px before its edge: 3X narrow px over the two narrow lenses' focal ratio
      d = np.minimum.reduce([nx, src.width - nx, ny, src.height - ny]) * lenses["os_narrow"]["f"] / lenses["ar_ox_narrow"]["f"]
      assert_close(alpha, np.where(n_ok, np.round(np.clip(d / feather, 0, 1) * 255), 0), ~(n_edge & (z > 0)))


@unittest.skipUnless(getenv("MODEL_PKL", ""), "requires an artifact from python examples/openpilot/compile_onnx.py")
class TestCompiledModel(unittest.TestCase):
  def setUp(self):
    self.artifact = load_pickle(getenv("MODEL_PKL", ""), out_of_band=bool(getenv("PICKLE_OOB")))
    self.model = self.artifact['run']

  def test_retarget(self):
    from tinygrad.runtime.autogen.libc import Elf64_Ehdr, EM_AARCH64
    make_retargetable(self.model)
    with Context(DEV=replace(DEV.target("CPU"), arch="arm64,generic")): linear = lower_and_compile(self.model.captured._linear)
    for p in linear.toposort(enter_calls=True):
      if p.op is Ops.PROGRAM and p.arg.target.device == "CPU":
        self.assertEqual(Elf64_Ehdr.from_buffer_copy(p.src[-1].arg).e_machine, EM_AARCH64)

  def test_inputs(self):
    def run(seed):
      inputs = make_inputs(self.artifact, seed)
      self.model(**inputs)
      return [t.numpy().copy() for t in inputs['output_buffers'].values()]
    original, changed, repeated = run(100), run(101), run(100)
    for before, after, again in zip(original, changed, repeated, strict=True):
      self.assertTrue(np.isfinite(before).all() and np.isfinite(after).all())
      np.testing.assert_array_equal(before, again)
    self.assertFalse(all(np.array_equal(a, b) for a, b in zip(original, changed, strict=True)))

  @unittest.skipUnless(getenv("MODEL_ONNX", ""), "requires the source ONNX model")
  def test_onnx(self):
    import onnxruntime as ort
    session = ort.InferenceSession(str(fetch(getenv("MODEL_ONNX", ""))), providers=['CPUExecutionProvider'])
    input_types = {v.name: v.type.removeprefix('tensor(').removesuffix(')') for v in session.get_inputs()}
    for seed in (100, 101):
      inputs = make_inputs(self.artifact, seed)
      reference = session.run(None, {k: v.numpy().astype({'float': 'float32', 'double': 'float64'}.get(input_types[k], input_types[k]))
                                    for k, v in inputs.items() if k in input_types})
      self.model(**inputs)
      for expected, actual in zip(reference, inputs['output_buffers'].values(), strict=True):
        fp16 = expected.dtype == np.float16
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-3 if fp16 else 1e-4,
                                   atol=5e-5 * float(np.abs(expected).max()) if fp16 else 1e-4)


if __name__ == '__main__': unittest.main()
