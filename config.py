#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py V3 — Fonte única de todas as constantes e configurações do pipeline.

CHANGELOG V3 (Modo Flux — campo de reator):
  - SimulationDefaults.DEPLETION_NORMALIZATION alterado para "flux".
    O wafer está imerso em campo de reator; o fluxo nominal do canal é
    prescrito diretamente ao IndependentOperator sem source_rate nem calibração.
  - SourceCalibrationConfig mantida por backward-compat mas com
    ENABLE_CALIBRATION=False por padrão.
  - GeometryContract: SOURCE_COVERS_FRONT_FACE removida (irrelevante no modo flux).
    Água mantida na geometria para cálculo correto de MicroXS pelo espectro moderado.
  - FluxModeConfig: nova classe com parâmetros do modo flux (reator).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "ValidationLimits",
    "PhysicsConstants",
    "TNLoopConfig",
    "CoolingConfig",
    "NuclearDataPaths",
    "GeometryLimits",
    "GeometryContract",
    "SourceCalibrationConfig",
    "FluxModeConfig",
    "ThermalSolverConfig",
    "ChainDataProxy",
    "SimulationDefaults",
    "SimulationModes",
    "DepletionAutoTuner",
    "DepletionParams",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES FÍSICAS
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsConstants:
    """Constantes físicas CODATA 2018 — imutáveis."""
    EV_TO_J: float = 1.602_176_634e-19   # J/eV  (exato)
    N_A:     float = 6.022_140_76e23     # mol⁻¹ (exato)
    KB_EV:   float = 8.617_333_262e-5    # eV/K  (kB em eV)
    LN2:     float = 0.693_147_180_559_945_3  # ln(2)
    SECONDS_PER_HOUR: float = 3600.0
    # Energia média por fissão de U235 [eV] — usada para cálculo de potência
    E_FISSION_U235_EV: float = 202.0e6   # ~202 MeV (inclui decaimento beta/gama)
    E_FISSION_EV:      float = 200.0e6   # valor conservador sem decaimentos


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIMITES DE VALIDAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationLimits:
    """Limites de segurança física."""
    TEMP_MIN_K:             float = 250.0
    TEMP_MAX_K:             float = 3500.0
    TEMP_COMBUSTIVEL_MAX_K: float = 3000.0
    TEMP_AGUA_MAX_K:        float = 623.0
    TEMP_ZIRCALOY_MAX_K:    float = 2263.0
    TEMP_ALUMINIO_MAX_K:    float = 855.0
    TEMP_ACA_MAX_K:         float = 1673.0
    RHO_MAX_GCM3:           float = 25.0
    SOURCE_RATE_MIN:        float = 1e3   # mantido por backward-compat


# ══════════════════════════════════════════════════════════════════════════════
# 2.5. CONTRATO GEOMÉTRICO
# ══════════════════════════════════════════════════════════════════════════════

class GeometryContract:
    """
    Parâmetros da geometria padrão.

    No modo flux (reator), a água ao redor do wafer é mantida para
    que o transporte MC de cálculo de MicroXS veja o espectro moderado
    correto — não há fonte plana nem SOURCE_PLANE.
    """
    # Mantidos por backward-compat com módulos que importam estas constantes
    DISTANCE_SOURCE_TO_FACE_CM: float = 1.0
    SOURCE_PLANE_THICKNESS_CM:  float = 1e-6
    SOURCE_DIRECTION:           Tuple[float, float, float] = (0.0, 0.0, 1.0)
    SOURCE_COVERS_FRONT_FACE:   bool = True  # irrelevante no modo flux


# ══════════════════════════════════════════════════════════════════════════════
# 2.6. CONFIGURAÇÃO DE CALIBRAÇÃO (desativada no modo flux)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceCalibrationConfig:
    """
    Parâmetros de calibração da fonte — DESATIVADA no modo flux.

    No modo flux, o fluxo nominal do canal é prescrito diretamente ao
    IndependentOperator (normalization_mode='flux'). Não há fonte plana
    nem source_rate a calibrar. Esta classe é mantida por backward-compat.
    """
    ENABLE_CALIBRATION:      bool  = False   # V3: desativado (modo flux)
    FLUX_TOLERANCE_REL:      float = 0.02
    MAX_ITERATIONS:          int   = 10
    PARTICLES_PER_ITERATION: int   = 50_000
    BATCHES_PER_ITERATION:   int   = 5
    UNDER_RELAXATION:        float = 0.8
    MIN_FLUX_MEASURED:       float = 1e-6
    USE_SURFACE_CURRENT:     bool  = False
    CALIBRATION_TALLY_NAME:  str   = "flux_calibration"
    RANDOM_SEED:             Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 < self.FLUX_TOLERANCE_REL < 1:
            raise ValueError(f"FLUX_TOLERANCE_REL fora de (0,1): {self.FLUX_TOLERANCE_REL}")
        if self.MAX_ITERATIONS < 1:
            raise ValueError(f"MAX_ITERATIONS < 1: {self.MAX_ITERATIONS}")
        if not 0 < self.UNDER_RELAXATION <= 1:
            raise ValueError(f"UNDER_RELAXATION fora de (0,1]: {self.UNDER_RELAXATION}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.7. CONFIGURAÇÃO DO MODO FLUX (reator)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FluxModeConfig:
    """
    Parâmetros do modo flux — IndependentOperator com fluxo prescrito.

    Uso: wafer imerso em campo de fluxo de reator com fluxo nominal medido
    ou especificado pelo operador do reator.

    IndependentOperator(materials, fluxes, chain_file, normalization_mode='flux')
      fluxes[i] = flux_nominal [n/cm²/s] para cada material depletável i
      O operador calcula MicroXS via MC (uma corrida) e depleta com Bateman.
    """
    # Fluxo uniforme em todas as camadas por padrão.
    # Se False, usa fluxo por camada de geometry_result (refinamento).
    UNIFORM_FLUX: bool = True

    # Tolerância para considerar material "depletável com fissão"
    # (usado para estimar potência via taxa de fissão)
    FISSILE_THRESHOLD: float = 1e-4   # fração mássica mínima de U/Pu

    # Número de partículas para corrida de cálculo de MicroXS
    # Maior que produção para boa estatística das seções de choque
    PARTICLES_MICROXS: int = 100_000

    # Se True, exporta MicroXS calculadas para arquivo JSON (diagnóstico)
    EXPORT_MICROXS: bool = True
    MICROXS_FILENAME: str = "microxs_calculated.json"


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIMITES GEOMÉTRICOS
# ══════════════════════════════════════════════════════════════════════════════

class GeometryLimits:
    NANOSCALE_MIN_CM: float = 1.0e-6
    GAP_TOLERANCE_CM: float = 1.0e-8


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOOP TÉRMICO-NEUTRÔNICO
# ══════════════════════════════════════════════════════════════════════════════

class TNLoopConfig:
    """Parâmetros de controle do acoplamento Térmico-Neutrônico (Phase F)."""

    ENABLE_TN_COUPLING:        bool  = False
    RELAXATION_FACTOR:         float = 0.5
    MAX_TN_ITERATIONS:         int   = 20
    CONVERGENCE_EPSILON_TEMP:  float = 0.5
    CONVERGENCE_EPSILON_POWER: float = 0.01
    CONVERGENCE_EPSILON_RHO:   float = 0.001
    MAX_TEMPERATURE_K:         float = 3500.0
    MIN_TEMPERATURE_K:         float = 273.0
    DYNAMIC_RELAXATION:        bool  = False
    PICARD_MAX_INTERNAL:       int   = 10

    @classmethod
    def from_parser(cls, parser_data: dict) -> None:
        sim_params = parser_data.get("simulation_parameters", {})
        raw = sim_params.get("thermal_coupling",
              sim_params.get("THERMAL_COUPLING", None))
        if raw is None:
            cls.ENABLE_TN_COUPLING = False
            return
        if isinstance(raw, bool):
            cls.ENABLE_TN_COUPLING = raw
        else:
            cls.ENABLE_TN_COUPLING = str(raw).strip().lower() in ("true","1","yes","sim")

    @classmethod
    def validate(cls) -> None:
        assert 0 < cls.RELAXATION_FACTOR < 1
        assert 0 < cls.CONVERGENCE_EPSILON_POWER < 1
        assert cls.MAX_TN_ITERATIONS >= 1
        assert cls.MIN_TEMPERATURE_K < cls.MAX_TEMPERATURE_K


# ══════════════════════════════════════════════════════════════════════════════
# 5. SOLVER TÉRMICO FDM
# ══════════════════════════════════════════════════════════════════════════════

class ThermalSolverConfig:
    H_CONV:     float = 5000.0
    PICARD_TOL: float = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# 6. RESFRIAMENTO PÓS-IRRADIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoolingConfig:
    COOLING_TIME_H:     float = 0.0
    COOLING_INTERVAL_H: float = 1.0
    ACTIVATE_STRUCTURAL: bool = False
    COMPUTE_DOSE:        bool = False

    def __post_init__(self) -> None:
        if self.COOLING_TIME_H < 0:
            raise ValueError(f"COOLING_TIME_H < 0: {self.COOLING_TIME_H}")
        if self.COOLING_INTERVAL_H <= 0:
            raise ValueError(f"COOLING_INTERVAL_H <= 0: {self.COOLING_INTERVAL_H}")
        if self.COOLING_TIME_H > 0 and self.COOLING_INTERVAL_H > self.COOLING_TIME_H:
            raise ValueError(
                f"COOLING_INTERVAL_H ({self.COOLING_INTERVAL_H}h) > "
                f"COOLING_TIME_H ({self.COOLING_TIME_H}h)"
            )

    @property
    def n_snapshots(self) -> int:
        if self.COOLING_TIME_H <= 0:
            return 0
        return max(1, round(self.COOLING_TIME_H / self.COOLING_INTERVAL_H)) + 1

    @property
    def phase_g_active(self) -> bool:
        return self.COOLING_TIME_H > 0

    @property
    def phase_h_active(self) -> bool:
        return self.ACTIVATE_STRUCTURAL

    @property
    def phase_i_active(self) -> bool:
        return self.COMPUTE_DOSE

    @classmethod
    def from_simulation_params(cls, sim_params: dict) -> "CoolingConfig":
        return cls(
            COOLING_TIME_H=float(sim_params.get("cooling_time_h", 0.0)),
            COOLING_INTERVAL_H=float(sim_params.get("cooling_interval_h", 1.0)),
            ACTIVATE_STRUCTURAL=bool(sim_params.get("activate_structural", False)),
            COMPUTE_DOSE=bool(sim_params.get("compute_dose", False)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. CAMINHOS DE DADOS NUCLEARES
# ══════════════════════════════════════════════════════════════════════════════

class NuclearDataPaths:
    _HOME = Path.home()
    _ND   = _HOME / "nuclear_data"

    XS_CANDIDATES: List[Tuple[str, Path]] = [
        ("ENDF-B-VIII.0", _ND / "endf_b_viii_0_hdf5"  / "cross_sections.xml"),
        ("JEFF-3.3",      _ND / "jeff33_hdf5"          / "cross_sections.xml"),
        ("TENDL-2021",    _ND / "hdf5_lib_tendl2021"   / "cross_sections.xml"),
        ("TENDL-2015",    _ND / "hdf5_lib_tendl2015"   / "cross_sections.xml"),
        ("ENDF-B-VII.1",  _ND / "endfb71_hdf5"         / "cross_sections.xml"),
    ]

    CHAIN_CANDIDATES: List[Path] = [
        Path("chain_endfb80_pwr.xml"),
        Path("chain_endfb80_act.xml"),
        _ND / "chain_endfb80_pwr.xml",
        _ND / "chain_endfb80_act.xml",
        _HOME / "chain_endfb80_pwr.xml",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 8. PROXY DE DADOS NUCLEARES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChainDataProxy:
    """Proxy leve para chain_file e xs_path — evita importar openmc em config."""
    chain_path: Optional[Path] = None
    xs_path:    Optional[Path] = None

    def get_chain_file(self) -> Optional[Path]:
        return self.chain_path if self.chain_path and self.chain_path.exists() else None

    def get_xs_path(self) -> Optional[Path]:
        return self.xs_path if self.xs_path and self.xs_path.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# 9. MODOS DE SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class SimulationModes:
    ACTIVATION:  str = "ACTIVATION"
    CRITICALITY: str = "CRITICALITY"
    BURNUP:      str = "BURNUP"
    AUTO:        str = "AUTO"
    ALL = {ACTIVATION, CRITICALITY, BURNUP, AUTO}

    @classmethod
    def is_valid(cls, mode: str) -> bool:
        return mode.upper() in cls.ALL

    @classmethod
    def needs_power(cls, mode: str) -> bool:
        return mode.upper() in {cls.BURNUP, cls.CRITICALITY}

    @classmethod
    def is_eigenvalue(cls, mode: str) -> bool:
        return mode.upper() == cls.CRITICALITY


# ══════════════════════════════════════════════════════════════════════════════
# 10. DEFAULTS DE SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class SimulationDefaults:
    # ── Temporal ─────────────────────────────────────────────────────────────
    DT_H:              float = 12.0
    DT_H_OUTPUT:       float = 12.0
    DT_H_DEPLETION:    float = 6.0
    TOTAL_TIME_H:      float = 48.0
    COOLING_TIME_H:    float = 6.0

    # ── Depleção ──────────────────────────────────────────────────────────────
    DEPLETION_INTEGRATOR:    str  = "celi"
    # V3: normalização por fluxo — modo reator, fluxo prescrito diretamente
    DEPLETION_NORMALIZATION: str  = "flux"
    DEPLETION_SUBSTEPS:      int  = 2

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    FLUX:           float = 1e13
    SOURCE_TEMP_K:  float = 300.0
    NPARTICLES:     int   = 100_000
    NBATCHES:       int   = 10
    NINACTIVE:      int   = 0
    NINACTIVE_EIGENVALUE: int = 50

    # ── Geometria ─────────────────────────────────────────────────────────────
    WAFER_SIDE_CM:  float = 1.69
    WATER_TEMP_C:   float = 25.0
    WATER_FLOW_M3S: float = 0.001
    POWER_W:        float = None


# ══════════════════════════════════════════════════════════════════════════════
# 11. AUTO-TUNER DE DEPLEÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class DepletionAutoTuner:
    """
    Escolhe DT_H_DEPLETION e DEPLETION_INTEGRATOR automaticamente.

    Tabela de decisão:
      DT_H ≤  6h  → DT_dep = DT/2, celi
      DT_H ≤ 12h  → DT_dep = DT/2, celi   (produção padrão)
      DT_H ≤ 24h  → DT_dep = DT/4, celi
      DT_H > 24h  → DT_dep = DT/6, si_celi
    """
    THRESHOLD_FINE:     float = 6.0
    THRESHOLD_MEDIUM:   float = 12.0
    THRESHOLD_COARSE:   float = 24.0

    DIVISOR_FINE:       int   = 2
    DIVISOR_MEDIUM:     int   = 2
    DIVISOR_COARSE:     int   = 4
    DIVISOR_VERYCOARSE: int   = 6

    INTEGRATOR_FINE:       str = "celi"
    INTEGRATOR_MEDIUM:     str = "celi"
    INTEGRATOR_COARSE:     str = "celi"
    INTEGRATOR_VERYCOARSE: str = "si_celi"

    DT_DEPLETION_MIN_H: float = 1.0

    @classmethod
    def tune(
        cls,
        dt_output_h:         float,
        user_dt_depletion_h: Optional[float] = None,
        user_integrator:     Optional[str]   = None,
    ) -> "DepletionParams":
        if dt_output_h <= cls.THRESHOLD_FINE:
            auto_dep, auto_integ, band = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_FINE), cls.INTEGRATOR_FINE, "fina"
        elif dt_output_h <= cls.THRESHOLD_MEDIUM:
            auto_dep, auto_integ, band = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_MEDIUM), cls.INTEGRATOR_MEDIUM, "média (produção)"
        elif dt_output_h <= cls.THRESHOLD_COARSE:
            auto_dep, auto_integ, band = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_COARSE), cls.INTEGRATOR_COARSE, "grossa"
        else:
            auto_dep, auto_integ, band = max(cls.DT_DEPLETION_MIN_H, dt_output_h / cls.DIVISOR_VERYCOARSE), cls.INTEGRATOR_VERYCOARSE, "muito grossa"

        final_dep   = user_dt_depletion_h if user_dt_depletion_h is not None else auto_dep
        final_integ = user_integrator      if user_integrator      is not None else auto_integ
        auto_tuned  = (user_dt_depletion_h is None or user_integrator is None)
        n_substeps  = max(1, round(dt_output_h / final_dep))

        return DepletionParams(
            dt_depletion_h=round(final_dep, 6),
            integrator=final_integ,
            n_substeps=n_substeps,
            band=band,
            auto_tuned=auto_tuned,
            dt_output_h=dt_output_h,
        )


@dataclass
class DepletionParams:
    dt_depletion_h: float
    integrador:     str   = ""
    integrator:     str   = ""
    n_substeps:     int   = 1
    band:           str   = ""
    auto_tuned:     bool  = True
    dt_output_h:    float = 0.0

    def __post_init__(self) -> None:
        if self.integrador and not self.integrator:
            self.integrator = self.integrador
        elif self.integrator and not self.integrador:
            self.integrador = self.integrator

    def log_summary(self, logger_fn) -> None:
        source = "auto-tune" if self.auto_tuned else "input usuário"
        logger_fn(
            "DepletionAutoTuner [%s, %s]: DT_output=%.1fh → "
            "DT_dep=%.1fh × %d sub-passos, integrador=%s",
            self.band, source,
            self.dt_output_h, self.dt_depletion_h,
            self.n_substeps, self.integrator,
        )
