import unittest
from unittest import mock

from medassistant import launcher


class TestGpuPatternClassification(unittest.TestCase):
    """dGPU-only heuristic (owner decision 2026-10-18): bare 'Arc' matches
    Intel Core Ultra *integrated* 'Arc Graphics' — require the A#/B# product
    letter. iGPUs (Intel 'Arc Graphics', Iris, Radeon 7xxM, bare Radeon)
    deliberately do NOT match."""

    def test_discrete_nvidia_matches(self):
        self.assertTrue(launcher.is_discrete_gpu("NVIDIA GeForce RTX 4060"))
        self.assertTrue(launcher.is_discrete_gpu("GeForce GTX 1080 Ti"))
        self.assertTrue(launcher.is_discrete_gpu("NVIDIA RTX A4000"))
        self.assertTrue(launcher.is_discrete_gpu("Quadro P4000"))

    def test_discrete_amd_matches(self):
        self.assertTrue(launcher.is_discrete_gpu("Radeon RX 6800 XT"))
        self.assertTrue(launcher.is_discrete_gpu("Radeon Pro W6600"))

    def test_discrete_arc_product_letter_matches(self):
        # Arc A# / Arc B# = discrete (Intel Arc discrete cards)
        self.assertTrue(launcher.is_discrete_gpu("Intel(R) Arc(TM) A770 Graphics"))
        self.assertTrue(launcher.is_discrete_gpu("Intel Arc B580"))

    def test_integrated_arc_graphics_does_not_match(self):
        # bare "Arc Graphics" is Intel Core Ultra iGPU -> CPU
        self.assertFalse(launcher.is_discrete_gpu("Intel(R) Arc(TM) Graphics"))
        self.assertFalse(launcher.is_discrete_gpu("Arc Graphics"))

    def test_iris_does_not_match(self):
        self.assertFalse(launcher.is_discrete_gpu("Intel(R) Iris(R) Xe Graphics"))
        self.assertFalse(launcher.is_discrete_gpu("Iris Plus Graphics 640"))

    def test_radeon_7xxm_bare_does_not_match(self):
        self.assertFalse(launcher.is_discrete_gpu("AMD Radeon 780M"))
        self.assertFalse(launcher.is_discrete_gpu("Radeon 660M"))
        self.assertFalse(launcher.is_discrete_gpu("Radeon Graphics"))

    def test_case_insensitive(self):
        self.assertTrue(launcher.is_discrete_gpu("nvidia geforce rtx 3060"))
        self.assertFalse(launcher.is_discrete_gpu("iris xe graphics"))


class TestDecideBackend(unittest.TestCase):
    def test_force_cpu_wins(self):
        with mock.patch.object(launcher, "adapter_names", return_value=["NVIDIA GeForce RTX 4060"]):
            self.assertEqual(launcher.decide_backend(force_cpu=True), "cpu")

    def test_force_gpu_wins(self):
        with mock.patch.object(launcher, "adapter_names", return_value=[]):
            self.assertEqual(launcher.decide_backend(force_gpu=True), "gpu")

    def test_force_cpu_and_gpu_exits(self):
        with self.assertRaises(SystemExit):
            launcher.decide_backend(force_cpu=True, force_gpu=True)

    def test_dgp_picks_gpu(self):
        with mock.patch.object(launcher, "adapter_names",
                               return_value=["NVIDIA GeForce RTX 4060"]):
            self.assertEqual(launcher.decide_backend(), "gpu")

    def test_igpu_only_picks_cpu(self):
        with mock.patch.object(launcher, "adapter_names",
                               return_value=["Intel(R) Arc(TM) Graphics"]):
            self.assertEqual(launcher.decide_backend(), "cpu")

    def test_no_adapter_picks_cpu(self):
        with mock.patch.object(launcher, "adapter_names", return_value=[]):
            self.assertEqual(launcher.decide_backend(), "cpu")


class TestAdapterNames(unittest.TestCase):
    def test_registry_read_path(self):
        fake_sub = "0000"
        fake_desc = "NVIDIA GeForce RTX 4060"
        with mock.patch("winreg.OpenKey") as mock_open, \
             mock.patch("winreg.EnumKey", side_effect=[fake_sub, OSError]), \
             mock.patch("winreg.QueryValueEx", return_value=(fake_desc, None)):
            names = launcher.adapter_names()
            self.assertIn(fake_desc, names)

    def test_no_winreg_returns_empty(self):
        with mock.patch.dict("sys.modules", {"winreg": None}):
            # adapter_names does `import winreg` inside; reload to simulate
            import importlib
            import medassistant.launcher as mod
            importlib.reload(mod)
            self.assertEqual(mod.adapter_names(), [])


if __name__ == "__main__":
    unittest.main()
