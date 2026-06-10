"""Cross-Backend consistency: the C++ CPU DNS and the Python 3D Solver run the
same Scenario with the same fixed dt and must agree on the energy history.

Needs cmake + FFTW3, so it is opt-in:  pytest -m cpp

Tolerance note: the Backends are not bit-identical by construction - the C++
dealiasing mask is spherical (|k|^2 <= k_cut^2, cpu_kernels.hpp) while the
Python one is a box (|k_i| <= k_cut).  For a short smooth TG run the affected
modes carry negligible energy, so the histories must still agree tightly.
"""

import re
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT
from ns3d import NS3D

pytestmark = pytest.mark.cpp

N = 32
NU = 0.02
DT = 1e-3
STEPS = 100        # = diag_interval, so the binary prints E_kin at t = 0.1


@pytest.fixture(scope="module")
def cpu_binary(tmp_path_factory):
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")
    build = tmp_path_factory.mktemp("build")
    cfg = subprocess.run(
        ["cmake", "-S", str(REPO_ROOT), "-B", str(build), f"-DNS3D_N={N}"],
        capture_output=True, text=True)
    if cfg.returncode != 0:
        pytest.skip(f"cmake configure failed (FFTW3 missing?):\n{cfg.stderr}")
    subprocess.run(["cmake", "--build", str(build), "-j"], check=True,
                   capture_output=True, text=True)
    return build / "ns3d_cpu"


def test_taylor_green_energy_matches_python(cpu_binary, tmp_path):
    out = subprocess.run(
        [str(cpu_binary), "--ic", "tg", "--nu", str(NU),
         "--dt", str(DT), "--steps", str(STEPS)],
        cwd=tmp_path,                      # keep output/ out of the repo
        capture_output=True, text=True, check=True)
    energies = [float(m) for m in
                re.findall(r"E_kin = ([0-9.eE+-]+)", out.stdout)]
    assert len(energies) >= 2, f"unexpected solver output:\n{out.stdout}"
    e0_cpp, e1_cpp = energies[0], energies[-1]

    sim = NS3D(N=N, nu=NU, scenario="tg")
    e0_py = sim.kinetic_energy()
    for _ in range(STEPS):
        sim.step(dt=DT)
    e1_py = sim.kinetic_energy()

    assert e0_py == pytest.approx(e0_cpp, rel=1e-12)   # identical IC
    assert e1_py == pytest.approx(e1_cpp, rel=1e-5)    # same method, see note
