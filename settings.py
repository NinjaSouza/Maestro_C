#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
settings.py V226 — Cria openmc.Settings e resolve bibliotecas de dados nucleares.

CHANGELOG V226 (Modo Flux — campo de reator):
  MUDANÇA PRINCIPAL: normalization_mode padrão alterado de 'source-rate' para 'flux'.
    O wafer está imerso num campo de fluxo de reator; o fluxo nominal do canal
    (FLUXO no input) é prescrito diretamente ao IndependentOperator sem
    source_rate nem calibração.

    IndependentOperator(materials, fluxes=[phi]*n_mats, chain_file, norm_mode='flux')
      fluxes[i] = FLUXO [n/cm²/s] para cada material depletável i
      OpenMC calcula MicroXS via MC (uma corrida) e depleta com Bateman analítico.

  REMOVIDO: source_rate, source_rate_initial, source_rates_initial, calibration_required.
    Esses campos eram necessários no modo source-rate; no modo flux são irrelevantes
    e sua presença causava confusão na passagem de parâmetros ao simulation.py.

  ADICIONADO: flux_per_material — lista de [n/cm²/s] com len = n_materiais depletáveis,
    pronta para passar como argumento 'fluxes' ao IndependentOperator.

  openmc.Settings: source removida no modo flux.
    A corrida de transporte para cálculo de MicroXS usa a geometria reflectiva
    definida em geometry.py — a fonte é implícita (boundary reflectiva simula
    campo isotrópico de reator).

  DepletionAutoTuner: sem mudanças — continua escolhendo dt_depletion_h e integrador.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import openmc
    _OPENMC_OK = True
except ImportError:
    _OPENMC_OK = False

from config import (
    ValidationLimits, NuclearDataPaths, ChainDataProxy,
    SimulationDefaults, PhysicsConstants, SimulationModes,
    DepletionAutoTuner,
)

_VL  = ValidationLimits()
_SD  = SimulationDefaults
_PC  = PhysicsConstants
logger = logging.getLogger(__name__)

_ChainDataProxy = ChainDataProxy


# ─────────────────────────────────────────────────────────────────────────────
# LibraryHierarchy
# ─────────────────────────────────────────────────────────────────────────────

class LibraryHierarchy:
    def __init__(self) -> None:
        self._available: List[Tuple[str, Path]] = []

    def discover(self) -> Tuple[Optional[str], Optional[Path]]:
        seen: set = set()
        for name, path in NuclearDataPaths.XS_CANDIDATES:
            if name in seen:
                continue
            seen.add(name)
            if path.exists():
                self._available.append((name, path))
                logger.info("✓ %s → %s", name, path)
            else:
                logger.warning("✗ %s não encontrado: %s", name, path)

        if not self._available:
            env_xs = os.environ.get("OPENMC_CROSS_SECTIONS")
            if env_xs and Path(env_xs).exists():
                self._available.append(("env:OPENMC_CROSS_SECTIONS", Path(env_xs)))
            else:
                logger.error("Nenhuma biblioteca de cross-sections encontrada!")
                return None, None

        name, path = self._available[0]
        logger.info("Biblioteca primária: %s", name)
        return name, path

    @property
    def available(self) -> List[Tuple[str, Path]]:
        return list(self._available)


# ─────────────────────────────────────────────────────────────────────────────
# SettingsBuilder
# ─────────────────────────────────────────────────────────────────────────────

class SettingsBuilder:
    VERSION = "V226"

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    @staticmethod
    def _norm(parser_data: dict, *keys, default=None):
        for k in keys:
            for src in (parser_data, parser_data.get("simulation_parameters", {})):
                v = src.get(k)
                if v is not None:
                    return v
        return default

    def _find_chain(self, parser_data: dict) -> Optional[Path]:
        inp = self._norm(parser_data, "chain_file", "chainfile", "CHAIN_FILE")
        if inp:
            for c in (Path(inp), Path.home() / "nuclear_data" / inp, Path.cwd() / inp):
                if c.exists():
                    logger.info("✓ Chain file (input): %s", c)
                    return c
            logger.warning("Chain '%s' declarado no input mas não encontrado", inp)
        for c in NuclearDataPaths.CHAIN_CANDIDATES:
            if c.exists():
                logger.info("✓ Chain file (auto): %s", c)
                return c
        logger.error("Chain file não encontrado. Defina CHAIN_FILE no input.")
        return None

    @staticmethod
    def _build_timesteps_depletion(
        dt_depletion_h: float,
        dt_output_h:    float,
        total_h:        float,
    ) -> Tuple[List[float], List[int]]:
        n_output  = max(1, round(total_h / dt_output_h))
        n_sub     = max(1, round(dt_output_h / dt_depletion_h))
        dt_sub_s  = (dt_output_h / n_sub) * _PC.SECONDS_PER_HOUR

        timesteps_s:    List[float] = []
        output_indices: List[int]   = []

        for i in range(n_output):
            is_last_output = (i == n_output - 1)
            for j in range(n_sub):
                is_last_sub = (j == n_sub - 1)
                if is_last_output and is_last_sub:
                    already_s   = sum(timesteps_s)
                    remaining_s = total_h * _PC.SECONDS_PER_HOUR - already_s
                    timesteps_s.append(max(remaining_s, dt_sub_s * 0.01))
                else:
                    timesteps_s.append(dt_sub_s)
                if is_last_sub:
                    output_indices.append(len(timesteps_s) - 1)

        logger.debug(
            "_build_timesteps_depletion: %d passos totais, %d pontos output "
            "(n_sub=%d × n_output=%d)",
            len(timesteps_s), len(output_indices), n_sub, n_output,
        )
        return timesteps_s, output_indices

    @staticmethod
    def _build_timesteps(dt_h: float, total_h: float) -> List[float]:
        n           = max(1, round(total_h / dt_h))
        remainder_h = total_h - (n - 1) * dt_h
        steps_h     = [dt_h] * (n - 1) + [remainder_h]
        return [s * _PC.SECONDS_PER_HOUR for s in steps_h]

    @staticmethod
    def _output_times_h(dt_h: float, total_h: float) -> List[float]:
        n     = max(1, round(total_h / dt_h))
        times = [round(i * dt_h, 10) for i in range(n + 1)]
        if abs(times[-1] - total_h) > 1e-6:
            times[-1] = total_h
        return times

    def build(self, parser_data: dict, geometry_result: dict) -> dict:
        sp   = parser_data.get("simulation_parameters", parser_data)
        meta = parser_data.get("metadata", {})

        def _get(*keys, default=None, cast=None):
            for k in keys:
                for src in (sp, meta, parser_data):
                    v = src.get(k)
                    if v is not None:
                        return cast(v) if cast else v
            return default

        # ── Parâmetros temporais ──────────────────────────────────────────
        dt_output_h = _get("dt_h", "DTH",      default=_SD.DT_H_OUTPUT,  cast=float)
        total_h     = _get("total_time_h", "TEMPO_TOTAL_H",
                            default=_SD.TOTAL_TIME_H, cast=float)
        cooling_h   = _get("cooling_time_h",   default=_SD.COOLING_TIME_H, cast=float)

        # ── Parâmetros MC ─────────────────────────────────────────────────
        particles = _get("nparticles", "neutrons_por_passo",
                          default=_SD.NPARTICLES, cast=int)
        batches   = _get("nbatches", "batches",
                          default=_SD.NBATCHES,   cast=int)
        inactive  = _get("ninactivebatches", "inactive_batches",
                          default=_SD.NINACTIVE,  cast=int)
        flux_n    = _get("flux", "fluxo",
                          default=_SD.FLUX, cast=float)

        sim_mode  = _get("simulation_mode", "SIMULATION_MODE",
                          default=SimulationModes.ACTIVATION)

        # ── Depleção — auto-tune ──────────────────────────────────────────
        user_dt_dep = _get("dt_h_depletion", "DTH_DEPLETION", default=None, cast=float)
        user_integ  = _get("depletion_integrator", "INTEGRADOR", default=None)

        dep_params = DepletionAutoTuner.tune(
            dt_output_h=dt_output_h,
            user_dt_depletion_h=user_dt_dep,
            user_integrator=user_integ,
        )
        dt_depletion_h = dep_params.dt_depletion_h
        dep_integrator = dep_params.integrator
        dep_params.log_summary(logger.info)

        # V226: normalização padrão = 'flux' (modo reator)
        dep_normalization = _get(
            "depletion_normalization", "NORMALIZACAO",
            default=_SD.DEPLETION_NORMALIZATION,  # 'flux' em config.py V3
        )
        if SimulationModes.needs_power(sim_mode):
            dep_normalization = "fission-q"
            logger.info("Modo %s: normalization forçada para 'fission-q'", sim_mode)

        # ── Geometria ─────────────────────────────────────────────────────
        wg   = geometry_result.get("wafer_geometry", {})
        x_cm = float(wg.get("x_cm", wg.get("xcm", _SD.WAFER_SIDE_CM)))
        y_cm = float(wg.get("y_cm", wg.get("ycm", _SD.WAFER_SIDE_CM)))

        # ── Número de materiais depletáveis ───────────────────────────────
        # Necessário para construir fluxes=[phi]*n_mats
        mats_list   = geometry_result.get("openmc_materials") or []
        if hasattr(mats_list, "__iter__"):
            dep_mats = [m for m in mats_list if getattr(m, "depletable", False)]
        else:
            dep_mats = []
        n_dep_mats = len(dep_mats) if dep_mats else 3   # fallback: 3 camadas

        # ── Dados nucleares ───────────────────────────────────────────────
        lib_hier = LibraryHierarchy()
        lib_name, xs_path = lib_hier.discover()
        if xs_path is None:
            return {"success": False, "error": "Nenhuma biblioteca XS encontrada"}

        chain_path = self._find_chain(parser_data)
        if chain_path is None:
            return {"success": False, "error": "Chain file não encontrado"}

        if _OPENMC_OK:
            openmc.config["cross_sections"] = str(xs_path)

        # ── V226: flux_per_material — pronto para IndependentOperator ─────
        # fluxes=[flux_n]*n_dep_mats é o argumento correto para:
        #   IndependentOperator(materials, fluxes, chain_file, norm_mode='flux')
        # Fisicamente: cada material depletável recebe o fluxo nominal do canal.
        flux_per_material = [float(flux_n)] * n_dep_mats
        logger.info(
            "Modo flux (reator): flux=%.4e n/cm²/s × %d materiais depletáveis",
            flux_n, n_dep_mats,
        )

        # ── openmc.Settings ───────────────────────────────────────────────
        if _OPENMC_OK:
            omc_settings = openmc.Settings()

            if SimulationModes.is_eigenvalue(sim_mode):
                omc_settings.run_mode = "eigenvalue"
                n_inactive = inactive if inactive > 0 else _SD.NINACTIVE_EIGENVALUE
                omc_settings.inactive = n_inactive
                logger.info("Modo eigenvalue: inactive=%d batches", n_inactive)
            else:
                # V226: modo flux — fixed source para cálculo de MicroXS
                # A geometria com boundary reflectiva (geometry.py V223) simula
                # o campo isotrópico do reator sem fonte explícita.
                omc_settings.run_mode = "fixed source"
                if inactive > 0:
                    omc_settings.inactive = inactive

                # V226: fonte isotrópica volumétrica na água
                # Representa o campo de nêutrons proveniente de todas as direções
                # no canal do reator — não há fonte plana nem colimação.
                wg_info = geometry_result.get("water_geometry", {})
                x_ext   = float(wg_info.get("total_x_cm", x_cm + 20.0)) / 2.0
                y_ext   = float(wg_info.get("total_y_cm", y_cm + 20.0)) / 2.0
                z_bot   = -float(wg_info.get("axial_cm", 5.0))
                z_top   = float(wg_info.get("total_z_cm",
                                  geometry_result.get("wafer_geometry", {})
                                  .get("total_thickness_cm", 0.2) + 10.0))

                source_box = openmc.stats.Box(
                    [-x_ext, -y_ext, z_bot],
                    [ x_ext,  y_ext, z_top],
                    only_fissionable=False,
                )
                # Distribuição isotrópica de ângulo (campo de reator)
                src_temp_k = _get("fonte_temperatura_k", "source_temp_k",
                                  default=300.0, cast=float)
                kT_eV = src_temp_k * _PC.KB_EV
                omc_settings.source = [openmc.IndependentSource(
                    space=source_box,
                    energy=openmc.stats.Maxwell(kT_eV),
                    # angle: isotrópico por padrão (sem Monodirectional)
                )]
                logger.info(
                    "Fonte isotrópica volumétrica: box=[%.1f,%.1f]×[%.1f,%.1f]×[%.2f,%.2f]cm "
                    "Maxwell(kT=%.4feV)",
                    -x_ext, x_ext, -y_ext, y_ext, z_bot, z_top, kT_eV,
                )

            omc_settings.particles = particles
            omc_settings.batches   = batches
            omc_settings.output    = {"summary": True}

            temp_treatment = _get("temperature_treatment", "temp_treatment",
                                   default="interpolation")
            temp_default_k = _get("temperature_default_k", "temp_default",
                                   default=294.0, cast=float)
            omc_settings.temperature = {
                "method":    temp_treatment,
                "default":   temp_default_k,
                "range":     [250.0, 2500.0],
                "tolerance": 200.0,
                "multipole": False,
            }
            logger.info("temperature: method=%s  default=%.1f K",
                        temp_treatment, temp_default_k)
        else:
            omc_settings = None

        # ── Timesteps ─────────────────────────────────────────────────────
        use_substeps = dt_depletion_h < dt_output_h - 1e-6
        if use_substeps:
            timesteps_s, output_indices = self._build_timesteps_depletion(
                dt_depletion_h, dt_output_h, total_h,
            )
            n_steps = len(timesteps_s)
            logger.info(
                "Sub-stepping ativo: %d passos internos (Δt=%.1fh) → %d pontos output",
                n_steps, dt_depletion_h, len(output_indices),
            )
        else:
            timesteps_s    = self._build_timesteps(dt_output_h, total_h)
            n_steps        = len(timesteps_s)
            output_indices = list(range(n_steps))
            logger.info("Timesteps: %d passos × %.2fh", n_steps, dt_output_h)

        output_times_h = self._output_times_h(dt_output_h, total_h)

        return {
            "success": True,
            "version": self.VERSION,

            "openmc_settings": omc_settings,

            "temporal_params": {
                "n_timesteps":       n_steps,
                "n_output_points":   len(output_indices),
                "dt_output_h":       dt_output_h,
                "dt_depletion_h":    dt_depletion_h,
                "total_time_h":      total_h,
                "cooling_time_h":    cooling_h,
                "output_indices":    output_indices,
                "output_times_h":    output_times_h,
                "chain_file":        str(chain_path),
                "n_timesteps_legacy": max(1, round(total_h / dt_output_h)),
                "dt_h":              dt_output_h,
            },

            "depletion_params": {
                "integrator":            dep_integrator,
                "normalization":         dep_normalization,
                "timesteps_s":           timesteps_s,
                "timesteps_internos_s":  timesteps_s if use_substeps else None,
                # V226: flux_per_material substitui source_rates
                # len = n_materiais depletáveis; unidade = n/cm²/s
                "flux_per_material":     flux_per_material,
                "use_substeps":          use_substeps,
                "n_substeps":            dep_params.n_substeps,
                "auto_tune_band":        dep_params.band,
                "auto_tuned":            dep_params.auto_tuned,
                "dt_depletion_h":        dt_depletion_h,
                # Mantido por backward-compat — agora é idêntico a flux_per_material
                # mas com unidade [n/s] interpretada como fluxo [n/cm²/s]
                # (nome legado; novo código deve usar flux_per_material)
                "source_rates":          flux_per_material,
            },

            "database_info": {
                "active_library":      lib_name,
                "xs_path":             str(xs_path),
                "available_libraries": [n for n, _ in lib_hier.available],
            },

            "timesteps": timesteps_s,

            "data_manager": ChainDataProxy(chain_path=chain_path, xs_path=xs_path),

            # V226: source_params simplificado — sem source_rate nem calibração
            "source_params": {
                "flux_n_cm2_s":       flux_n,
                "flux_per_material":  flux_per_material,
                "n_dep_materials":    n_dep_mats,
                "wafer_area_cm2":     x_cm * y_cm,
                "normalization_mode": dep_normalization,
                # calibration_required=False: modo flux não usa calibração
                "calibration_required": False,
            },

            "simulation_mode": sim_mode,
            "error": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def create_settings(
    geometry_result:   Optional[Dict] = None,
    materials_list:    Optional[list] = None,
    simulation_params: Optional[Dict] = None,
    energy_source:     Any            = None,
    debug:             bool           = False,
    **kwargs,
) -> dict:
    _is_sig_b = (
        isinstance(geometry_result, dict)
        and isinstance(materials_list, dict)
        and "openmc_geometry"  in materials_list
        and "openmc_materials" in materials_list
    )
    if _is_sig_b:
        return SettingsBuilder(debug=debug).build(geometry_result, materials_list)

    _geometry_result = geometry_result or {}
    _sim_p           = simulation_params or {}
    _parser_data = {
        "simulation_parameters": _sim_p,
        "chain_file":            _sim_p.get("chain_file",           _sim_p.get("chainfile", "")),
        "flux":                  _sim_p.get("flux",                 _sim_p.get("fluxo", 1e13)),
        "dt_h":                  _sim_p.get("dt_h",                 _sim_p.get("dth", _SD.DT_H_OUTPUT)),
        "dt_h_depletion":        _sim_p.get("dt_h_depletion",       _SD.DT_H_DEPLETION),
        "total_time_h":          _sim_p.get("total_time_h",         _sim_p.get("totaltimeh", _SD.TOTAL_TIME_H)),
        "cooling_time_h":        _sim_p.get("cooling_time_h",       _SD.COOLING_TIME_H),
        "nparticles":            _sim_p.get("nparticles",           _sim_p.get("neutrons_por_passo", _SD.NPARTICLES)),
        "nbatches":              _sim_p.get("nbatches",             _sim_p.get("batches", _SD.NBATCHES)),
        "ninactivebatches":      _sim_p.get("ninactivebatches",     _sim_p.get("inactive_batches", _SD.NINACTIVE)),
        "fonte_temperatura_k":   _sim_p.get("fonte_temperatura_k",  _SD.SOURCE_TEMP_K),
        "simulation_mode":       _sim_p.get("simulation_mode",      SimulationModes.ACTIVATION),
        "depletion_integrator":  _sim_p.get("depletion_integrator", _SD.DEPLETION_INTEGRATOR),
        "depletion_normalization": _sim_p.get("depletion_normalization", _SD.DEPLETION_NORMALIZATION),
    }
    return SettingsBuilder(debug=debug).build(_parser_data, _geometry_result)
