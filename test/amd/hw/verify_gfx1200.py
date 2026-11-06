# Writes test/mockgpu/amd/gfx1200_verified.sigs: the rdna4 instruction signatures whose emulator results were bit-identical to a gfx1200
# gpu in every probe that ran them.
#   python test/amd/hw/verify_gfx1200.py unpack HW_ROOT    the committed gfx1200 results (test/mockgpu/amd/gfx1200_hw/*.part*) -> HW_ROOT
#   python test/amd/hw/verify_gfx1200.py emu HW_ROOT EMU_ROOT    every probe on the emulator (DEV=MOCKUSB+AMD), same signatures as HW_ROOT
#   python test/amd/hw/verify_gfx1200.py verify HW_ROOT EMU_ROOT
# A root holds all/ mods/ sigs/ consts/ dpp/ (probe_all.py), mat/ (probe_matrix.py) and rounding.npz (probe_rounding.py). A signature is
# listed only if it matched everywhere it ran: an e32 16-bit op also needs its |hi16 variant, a dual op both of its x/y slots, v_div_scale
# its aliased forms.
import sys, os, glob, json, tarfile, tempfile, subprocess, collections
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(here, "..", "..", "mockgpu", "amd", "gfx1200_hw")
PASSES = {"all": "run", "mods": "mods", "consts": "consts", "dpp": "dpp"}
ROUNDING = {"rcp_f32": "v_rcp_f32_e32", "sqrt_f32": "v_sqrt_f32_e32", "exp_f32": "v_exp_f32_e32", "rsq_f32": "v_rsq_f32_e32",
            "log_f32": "v_log_f32_e32", "rcp_f16": "v_rcp_f16_e32", "exp_f16": "v_exp_f16_e32", "sqrt_f16": "v_sqrt_f16_e32",
            "rsq_f16": "v_rsq_f16_e32", "log_f16": "v_log_f16_e32",
            "fma_f16": "v_fma_f16", "fma_mixlo_f16": "v_fma_mixlo_f16", "cvt_f16_f32": "v_cvt_f16_f32_e32",
            "wmma_f32_16x16x16_f16": "v_wmma_f32_16x16x16_f16|opsel_hi=3|opsel_hi2=1"}
FOLD = ("|hi16", "|x", "|y", "|s0=s1", "|s0=s2", "|consts", "|dpp", "|finite")

def _same(a, b) -> bool: return a.shape == b.shape and np.array_equal(a, b)

def unpack(out:str):
  parts = sorted(glob.glob(os.path.join(PARTS, "gfx1200_hw.tar.part*")))
  os.makedirs(out, exist_ok=True)
  with tempfile.TemporaryFile() as f:
    for p in parts:
      with open(p, "rb") as g: f.write(g.read())
    f.seek(0)
    with tarfile.open(fileobj=f) as t: t.extractall(out, filter="data")

def emu(hw:str, out:str):
  env = {**os.environ, "DEV": "MOCKUSB+AMD", "CHECK_OOB": "0", "GMMU": "0"}
  def probe(*a): subprocess.run([sys.executable, *a], env=env, check=True, stdout=subprocess.DEVNULL)
  for d, mode in PASSES.items(): probe(os.path.join(here, "probe_all.py"), mode, os.path.join(out, d))
  sigs = sorted(os.path.basename(f)[:-4].replace("+", "|") for f in glob.glob(os.path.join(hw, "sigs", "*.npz")))
  with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(sigs, f)
  probe(os.path.join(here, "probe_all.py"), "sigs", os.path.join(out, "sigs"), f.name)
  probe(os.path.join(here, "probe_matrix.py"), "run", os.path.join(out, "mat"))
  probe(os.path.join(here, "probe_rounding.py"), "run", os.path.join(out, "rounding.npz"))

def dir_results(hw_dir, emu_dir, res, key_of):
  for f in sorted(glob.glob(os.path.join(hw_dir, "*.npz"))):
    a, g = np.load(f), os.path.join(emu_dir, os.path.basename(f))
    if "error" in a.files: continue
    b = np.load(g) if os.path.exists(g) else None
    ok = b is not None and "error" not in b.files and int(a["inhash"]) == int(b["inhash"]) and _same(a["out"], b["out"])
    res[key_of(os.path.basename(f)[:-4], a)].append(ok)

def verify(hw:str, emu_root:str):
  res: dict[str, list[bool]] = collections.defaultdict(list)
  a, b = np.load(os.path.join(hw, "rounding.npz")), np.load(os.path.join(emu_root, "rounding.npz"))
  for k in a.files:
    sig = next((s for p, s in ROUNDING.items() if k == p or k.startswith(p + "_")), None)
    if sig is not None: res[sig].append(k in b.files and _same(a[k], b[k]))
  for d in (*PASSES, "sigs"): dir_results(os.path.join(hw, d), os.path.join(emu_root, d), res, lambda name, _: name.replace("+", "|"))
  dir_results(os.path.join(hw, "mat"), os.path.join(emu_root, "mat"), res, lambda name, a: str(a["sig"]))
  folded: dict[str, list[bool]] = collections.defaultdict(list)
  for k, oks in res.items(): folded[k.rsplit("|", 1)[0] if k.endswith(FOLD) else k] += oks
  ok = sorted(s for s, oks in folded.items() if oks and all(oks))
  bad = sorted(s for s, oks in folded.items() if not all(oks))
  out = os.path.join(here, "..", "..", "mockgpu", "amd", "gfx1200_verified.sigs")
  with open(out, "w") as f:
    f.write("# rdna4 instruction signatures the amd emulator reproduces bit for bit on a gfx1200 gpu; written by test/amd/hw/verify_gfx1200.py\n")
    f.write("".join(s + "\n" for s in ok))
  print(f"{len(ok)} verified signatures -> {os.path.normpath(out)}; {len(bad)} not verified")
  return bad

if __name__ == "__main__":
  if sys.argv[1] == "unpack": unpack(sys.argv[2])
  elif sys.argv[1] == "emu": emu(sys.argv[2], sys.argv[3])
  elif sys.argv[1] == "verify":
    for s in verify(sys.argv[2], sys.argv[3]): print("  not verified:", s)
