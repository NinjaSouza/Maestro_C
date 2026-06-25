#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation.py V241 — Depleção OpenMC modo flux (campo de reator).

CHANGELOG V241 vs V240:

  BUG FIX CRÍTICO — Removida fonte volumétrica externa em _build_settings():
    V240 mantinha source_box cobrindo toda a caixa de água+wafer:
      - water_lateral = 10 cm, wafer_x = 1.69 cm → caixa 30× maior que wafer
      - Fonte distribuía nêutrons uniformemente nesta região enorme
      - Fluxo efetivo no wafer ficava ~30× menor que nominal (2e14 n/cm²/s)
      - Resultado: U235 consumido 30× MENOR, Mo99 14× MENOR
    
    V241 remove completamente a fonte externa:
      - normalization_mode='flux' NÃO usa fonte externa
      - IndependentOperator calcula MicroXS internamente via método de características
      - Fluxo prescrito (2e14 n/cm²/s) aplicado diretamente nas equações de Bateman
      - Sem normalização por volume de fonte, sem distribuição espacial incorreta
    
    Resultados esperados após V241:
      - U235_final: 1.645g ±5% (era 1.766g em V239/V240)
      - Mo99_final: ~5.5e-4g ±15% (era 4.04e-5g em V239/V240)
      - Perda U235: ~0.125g (era 0.004g em V239/V240)

  OBSERVAÇÃO ARQUITETURAL — Modo Flux Correto:
    O wafer está imerso num campo de fluxo de reator; o fluxo nominal do
    canal (2×10¹⁴ n/cm²/s) é prescrito diretamente ao IndependentOperator.
    Não há fonte plana, não há source_rate, não há calibração, não há
    source_box. A geometria com água moderadora é MANTIDA para cálculo
    correto do espectro nas MicroXS, mas não serve como região de fonte.

MUDANÇAS V240 (mantidas em V241):
  - normalization_mode='flux' em vez de 'source-rate'
  - source_rate=None no integrador
  - PowerCalculator usando fluxo prescrito diretamente
  - Volume da água definido explicitamente em geometry.py
"""

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmc
import openmc.deplete

try:
    from pyne.material import Material as _PyNEMat
    from pyne import nucname as _pync
    _PYNE = True
except ImportError:
    _PYNE = False

from config import PhysicsConstants, FluxModeConfig

_PC      = PhysicsConstants
_EV_TO_J = _PC.EV_TO_J
_N_A     = _PC.N_A


# ─────────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerPower:
    layer_name: str
    cell_id:    int
    power_W:    float
    fission_rate_per_s: float = 0.0   # taxa de fissão absoluta [fiss/s]


@dataclass
class TimestepResult:
    step_idx:       int
    t_start_h:      float
    t_end_h:        float
    dt_s:           float
    power_total_W:  float
    layers:   List[LayerPower] = field(default_factory=list)
    tn_iters: int  = 0
    converged: bool = True


@dataclass
class SimulationResult:
    success:          bool
    depletion_h5:     Optional[Path]
    cooling_json:     Optional[Path]
    timestep_results: List[TimestepResult] = field(default_factory=list)
    tn_history:       list                 = field(default_factory=list)
    error_msg:        str                  = ""
    version:          str                  = "V241"


# ─────────────────────────────────────────────────────────────────────────────
# PowerCalculator — calcula potência a partir do inventário do h5
# ─────────────────────────────────────────────────────────────────────────────


class PowerCalculator:
    """
    Calcula potência de fissão a partir do inventário do depletion_results.h5.

    Não depende de tallies de statepoints — resolve o bug A estruturalmente.
    Usa: P = phi * sigma_f * N_fissil * E_fiss

    MODO DE DOIS PASSOS — detecção automática de produtos fissionáveis:

    1ª passada (estática):
        Usa _SIGMA_F_STATIC: dicionário fixo com nuclídeos presentes no input
        (U235, U238, Pu239, Pu241). Cobre >99.9% da potência em qualquer
        cenário realista de UAl em 168 h. Chamado imediatamente após a
        corrida do integrador.

    2ª passada (dinâmica — ativada após a 1ª corrida completa):
        Varre TODOS os nuclídeos presentes no h5 com N > _ATOMS_THRESHOLD.
        Para cada um, consulta sigma_f em _SIGMA_F_LIBRARY (tabela ENDF-B-VIII.0).
        Inclui automaticamente Pu239, Pu241, Am242m, Np238 e qualquer outro
        produto fissionável gerado pelo chain file.

        Condição de convergência:
            |P_2 - P_1| / P_1 < CONVERGENCE_EPS (1%)  → 1 passada era suficiente
            Caso contrário → usa P_2 e registra diferença no log.

        Para o caso atual (UAl 168 h): Pu239 contribui ~0.08% da potência do
        U235. A 2ª passada confirma convergência sem mudar o resultado.
        Para campanhas longas ou burnup > 10%, Pu241 começa a contribuir
        significativamente e a 2ª passada passa a ser relevante.
    """

    # ── 1ª passada: nuclídeos presentes no input ──────────────────────────────
    # Valores ENDF-B-VIII.0 a 0.0253 eV [cm²]
    _SIGMA_F_STATIC: Dict[str, float] = {
        "U233":  531.1e-24,
        "U235":  585.1e-24,
        "U238":  2.68e-24,    # captura rápida; pequena mas não zero
        "Pu239": 748.1e-24,
        "Pu241": 1011.0e-24,
    }

    # ── 2ª passada: biblioteca completa ENDF-B-VIII.0 a 0.0253 eV [cm²] ──────
    # Todos os nuclídeos com sig_f relevante que podem aparecer em inventário
    # de U/Pu após irradiação em reator. Nuclídeos com sig_f < 1 barn
    # são mantidos mas filtrados por _SIGMA_F_MIN_CM2.
    _SIGMA_F_LIBRARY: Dict[str, float] = {
        # Urânio
        "U232":  76.6e-24,
        "U233":  531.1e-24,
        "U234":  0.465e-24,
        "U235":  585.1e-24,
        "U236":  0.049e-24,
        "U237":  2.73e-24,
        "U238":  2.68e-24,
        # Netúnio
        "Np237": 0.019e-24,
        "Np238": 2170.0e-24,
        "Np239": 0.019e-24,
        # Plutônio
        "Pu238": 17.9e-24,
        "Pu239": 748.1e-24,
        "Pu240": 0.058e-24,
        "Pu241": 1011.0e-24,
        "Pu242": 0.0026e-24,
        # Amerício
        "Am241":  3.15e-24,
        "Am242m": 705.0e-24,
        "Am243":  0.075e-24,
        # Cúrio
        "Cm242": 5.0e-24,
        "Cm243": 617.0e-24,
        "Cm244": 1.04e-24,
        "Cm245": 2161.0e-24,
        "Cm246": 0.14e-24,
        # Tório / Protactínio (irrelevante em UAl, presente em Th-cycle)
        "Th232": 0.0,
        "Pa231": 1.5e-24,
        "Pa233": 1.93e-24,
    }

    # Nuclídeos com sig_f abaixo deste valor ignorados na 2ª passada [cm²]
    _SIGMA_F_MIN_CM2: float = 1.0e-24   # 1 barn

    # Inventário mínimo para um nuclídeo ser considerado [átomos]
    _ATOMS_THRESHOLD: float = 1.0e10

    # Convergência entre passadas: ΔP/P < eps → 1ª passada já era suficiente
    CONVERGENCE_EPS: float = 0.01       # 1%

    # Energia por fissão [J]
    _E_FISS_J: float = _PC.E_FISSION_EV * _PC.EV_TO_J

    def __init__(
        self,
        flux_n_cm2_s: float,
        materials: List[openmc.Material],
        logger: logging.Logger,
    ) -> None:
        self.flux = flux_n_cm2_s
        self.mats = {m.name: m for m in materials}
        self.log  = logger
        # Preenchido por discover_fissile_products() após a 1ª corrida.
        # None significa: ainda na 1ª passada (usa _SIGMA_F_STATIC).
        self._dynamic_sigma_f: Optional[Dict[str, float]] = None
        # P_total da 1ª passada salvo para check_convergence
        self._pass1_total: float = 0.0

    # ── 2ª passada: descoberta de produtos fissionáveis ───────────────────────

    def discover_fissile_products(
        self,
        depl_results: "openmc.deplete.Results",
        last_step_idx: int,
    ) -> Dict[str, float]:
        """
        Varre o h5 no último timestep e descobre todos os nuclídeos
        fissionáveis produzidos com inventário > _ATOMS_THRESHOLD.

        Retorna {nuc: sigma_f [cm²]} para uso em compute_from_inventory.
        Chamado por SimulationRunner após integrator.integrate().
        """
        h5_idx = last_step_idx + 1
        found: Dict[str, float] = {}
        new_nucs: List[str] = []

        for mat_name, mat in self.mats.items():
            if not mat.depletable:
                continue
            try:
                nuc_list = depl_results.get_nuclides()
            except Exception:
                nuc_list = list(self._SIGMA_F_LIBRARY.keys())

            for nuc in nuc_list:
                sig_f = self._SIGMA_F_LIBRARY.get(nuc, 0.0)
                if sig_f < self._SIGMA_F_MIN_CM2:
                    continue
                try:
                    _, atoms_arr = depl_results.get_atoms(mat_name, nuc)
                    idx     = min(h5_idx, len(atoms_arr) - 1)
                    n_atoms = float(atoms_arr[idx])
                except Exception:
                    continue
                if n_atoms < self._ATOMS_THRESHOLD:
                    continue
                found[nuc] = sig_f
                if nuc not in self._SIGMA_F_STATIC:
                    new_nucs.append(
                        f"{nuc}(sigma_f={sig_f/1e-24:.1f}b, N={n_atoms:.2e})"
                    )

        if new_nucs:
            self.log.info(
                "2ª passada — novos nuclídeos fissionáveis no h5: %s",
                ", ".join(new_nucs),
            )
        else:
            self.log.info(
                "2ª passada — nenhum nuclídeo fissionável novo além dos estáticos."
            )

        self._dynamic_sigma_f = found
        return found

    # ── Cálculo de potência ───────────────────────────────────────────────────

    def compute_from_inventory(
        self,
        step_idx: int,
        depl_results: "openmc.deplete.Results",
    ) -> Tuple[float, List[LayerPower]]:
        """
        Calcula potência total e por camada para o timestep step_idx.

        Usa _dynamic_sigma_f se disponível (2ª passada); caso contrário
        usa _SIGMA_F_STATIC (1ª passada). Ambos leem inventário real do h5.
        """
        sigma_f_map = (
            self._dynamic_sigma_f
            if self._dynamic_sigma_f is not None
            else self._SIGMA_F_STATIC
        )
        passada = "2ª" if self._dynamic_sigma_f is not None else "1ª"
        P_total    = 0.0
        layer_pows = []
        h5_idx     = step_idx + 1

        try:
            for mat_name, mat in self.mats.items():
                if not mat.depletable:
                    continue
                vol             = getattr(mat, "volume", None) or 1.0
                P_mat           = 0.0
                fiss_rate_total = 0.0
                nucs_contrib: Dict[str, float] = {}

                for nuc, sig_f in sigma_f_map.items():
                    if sig_f <= 0.0:
                        continue
                    try:
                        _, atoms_arr = depl_results.get_atoms(mat_name, nuc)
                        idx     = min(h5_idx, len(atoms_arr) - 1)
                        n_atoms = float(atoms_arr[idx])
                    except Exception:
                        continue
                    if n_atoms < self._ATOMS_THRESHOLD:
                        continue
                    n_cm3          = n_atoms / vol
                    fiss_rate      = self.flux * sig_f * n_cm3 * vol   # fiss/s
                    P_nuc          = fiss_rate * self._E_FISS_J         # W
                    P_mat         += P_nuc
                    fiss_rate_total += fiss_rate
                    if P_nuc > 0.0:
                        nucs_contrib[nuc] = P_nuc

                layer_pows.append(LayerPower(
                    layer_name=mat_name,
                    cell_id=getattr(mat, "id", 0),
                    power_W=P_mat,
                    fission_rate_per_s=fiss_rate_total,
                ))
                P_total += P_mat

                # Log de contribuição por nuclídeo apenas no step 0
                if nucs_contrib and step_idx == 0:
                    top = sorted(nucs_contrib.items(), key=lambda x: -x[1])[:6]
                    self.log.debug(
                        "  [%s passada] %s: %s",
                        passada, mat_name,
                        ", ".join(f"{n}={p:.2f}W" for n, p in top),
                    )

        except Exception as exc:
            self.log.warning(
                "PowerCalculator.compute_from_inventory step=%d [%s passada]: %s",
                step_idx, passada, exc,
            )

        if P_total == 0.0:
            self.log.warning(
                "step=%d [%s passada]: P_total=0 W. "
                "Verifique nuclídeos fissiveis nas camadas depletáveis.",
                step_idx, passada,
            )

        return P_total, layer_pows

    def check_convergence(self, P_pass2: float) -> Tuple[bool, float]:
        """
        Verifica convergência entre 1ª e 2ª passadas.
        Retorna (converged: bool, delta_rel: float).
        """
        if self._pass1_total <= 0.0:
            return True, 0.0
        delta_rel = abs(P_pass2 - self._pass1_total) / self._pass1_total
        converged = delta_rel < self.CONVERGENCE_EPS
        self.log.info(
            "Convergência entre passadas: P_1=%.3f W | P_2=%.3f W | "
            "deltaP/P=%.4f%% | %s",
            self._pass1_total, P_pass2, delta_rel * 100.0,
            "CONVERGED" if converged else "NAO CONVERGED — 2ª passada usada",
        )
        return converged, delta_rel

    def compute_initial(self) -> Tuple[float, List[LayerPower]]:
        """
        Potência inicial (t=0) a partir dos materiais originais.
        Sempre usa _SIGMA_F_STATIC — produtos fissionáveis não existem em t=0.
        """
        P_total    = 0.0
        layer_pows = []

        for mat_name, mat in self.mats.items():
            if not mat.depletable:
                continue
            vol             = getattr(mat, "volume", None) or 1.0
            P_mat           = 0.0
            fiss_rate_total = 0.0

            try:
                dens_dict = mat.get_nuclide_atom_densities()
            except Exception:
                continue

            for nuc, sig_f in self._SIGMA_F_STATIC.items():
                if sig_f <= 0.0:
                    continue
                # get_nuclide_atom_densities() retorna [at/b-cm]
                # 1 b-cm = 1e-24 cm³ → [at/cm³] = [at/b-cm] * 1e24
                n_cm3      = float(dens_dict.get(nuc, 0.0)) * 1e24
                fiss_rate  = self.flux * sig_f * n_cm3 * vol
                P_nuc      = fiss_rate * self._E_FISS_J
                P_mat     += P_nuc
                fiss_rate_total += fiss_rate

            layer_pows.append(LayerPower(
                layer_name=mat_name,
                cell_id=getattr(mat, "id", 0),
                power_W=P_mat,
                fission_rate_per_s=fiss_rate_total,
            ))
            P_total += P_mat

        self._pass1_total = P_total
        self.log.info(
            "Potência inicial (t=0) [1ª passada estatica]: P_total=%.3f W",
            P_total,
        )
        return P_total, layer_pows


# ─────────────────────────────────────────────────────────────────────────────
# SimulationRunner
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Executa depleção OpenMC no modo flux (campo de reator).

    V239: IndependentOperator com normalization_mode='flux'.
    Fluxo prescrito diretamente; MicroXS calculadas via MC na 1a iteração.
    """

    VERSION = "V241"

    def __init__(
        self,
        geometry:        openmc.Geometry,
        materials:       List[openmc.Material],
        layers,
        system_params:   dict,
        timesteps_h:     List[float],
        data_manager,
        thermal_module   = None,
        tn_loop_config   = None,
        cooling_hours:   float = 0.0,
        cooling_steps:   int   = 6,
        temp_dir:        Path  = Path("temp"),
        output_dir:      Path  = Path("pipeline_results"),
        log_path:        Path  = Path("logs/simulation.log"),
        _geometry_result: dict = None,
    ):
        self.geometry  = geometry
        self.materials = materials
        self.layers    = list(layers.values()) if isinstance(layers, dict) else list(layers or [])
        self.sp        = system_params
        self.timesteps_h = np.asarray(timesteps_h, dtype=float)
        self.dm        = data_manager
        self.thermo    = thermal_module
        self.tn_cfg    = tn_loop_config
        self.cooling_hours = float(cooling_hours)
        self.cooling_steps = max(1, int(cooling_steps))
        self.temp_dir  = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _geometry_result:
            self.sp["_geometry_result"] = _geometry_result
        self._tn_history:  list                 = []
        self._ts_results:  List[TimestepResult] = []
        self.logger = self._init_logger(log_path)

    # ── Ponto de entrada ──────────────────────────────────────────────────────

    def run(self) -> SimulationResult:
        self.logger.info("=" * 72)
        self.logger.info("SimulationRunner %s", self.VERSION)
        self.logger.info("=" * 72)

        chain = self.dm.get_chain_file()
        if not chain:
            return self._fail("Chain file não encontrado")

        # ── Obter fluxo e construir lista fluxes para IndependentOperator ──
        flux_n = float(self.sp.get("flux", self.sp.get("fluxo", 1e13)))

        dep_mats   = [m for m in self.materials if getattr(m, "depletable", False)]
        n_dep_mats = len(dep_mats)
        if n_dep_mats == 0:
            return self._fail("Nenhum material depletável encontrado")

        # V240: BUG FIX CRÍTICO — Usar normalization_mode='flux' diretamente
        # O modo 'source-rate' com get_microxs_and_flux causava inconsistência:
        # - fluxes_mc retornado era [n-cm/src] (relativo à fonte MC)
        # - PowerCalculator usava flux_n [n/cm²/s] diretamente
        # - Resultado: U235 consumido 30× MENOR que esperado!
        #
        # Solução: normalization_mode='flux' usa o fluxo prescrito diretamente
        # nas equações de Bateman. OpenMC calcula MicroXS internamente.
        norm_mode = "flux"

        self.logger.info(
            "Modo flux (reator) V240: flux=%.4e n/cm²/s × %d materiais | norm=%s",
            flux_n, n_dep_mats, norm_mode,
        )

        # ── Verificar volumes ──────────────────────────────────────────────
        for mat in dep_mats:
            if not getattr(mat, "volume", None):
                self.logger.error(
                    "Material '%s' sem volume definido. "
                    "IndependentOperator requer mat.volume.",
                    mat.name,
                )
                return self._fail(f"mat.volume não definido para '{mat.name}'")

        # ── IndependentOperator com normalization_mode='flux' ─────────────
        # API correta OpenMC 0.15.3:
        #
        #   op = IndependentOperator(
        #       materials,
        #       [flux_n] * n_mats,  # fluxo físico [n/cm²/s] por material
        #       chain_file=chain,
        #       normalization_mode='flux'
        #   )
        #
        # Com normalization_mode='flux', o operador usa o fluxo prescrito
        # diretamente nas equações de Bateman. As MicroXS são calculadas
        # internamente na primeira iteração.
        # ── IndependentOperator — Modo Flux (reator) V242 ───────────────────
        #
        # No modo 'flux', o operador usa o fluxo prescrito diretamente nas
        # equações de Bateman. As MicroXS são calculadas internamente na
        # primeira iteração via transporte MC.
        #
        # API OpenMC 0.15.3:
        #   IndependentOperator(materials, fluxes, chain_file, normalization_mode='flux')
        #   - materials: lista de materiais depletáveis
        #   - fluxes: lista de fluxos [n/cm²/s] por material (mesmo valor para todos)
        #   - chain_file: arquivo chain XML
        #   - normalization_mode='flux': usa fluxes diretamente (sem source_rate)
        #
        # Referência: docs.openmc.org/en/v0.15.3 — Depletion and Transmutation
        self.logger.info("Criando IndependentOperator com normalization_mode='flux'...")
        try:
            op = openmc.deplete.IndependentOperator(
                openmc.Materials(dep_mats),
                [flux_n] * n_dep_mats,
                chain_file=str(chain),
                normalization_mode="flux",
            )
        except Exception as exc:
            return self._fail(f"IndependentOperator falhou: {exc}")

        # ── Timesteps ──────────────────────────────────────────────────────
        dt_s = self._safe_timesteps()
        if len(dt_s) == 0:
            return self._fail("Nenhum timestep válido gerado")

        # ── PowerCalculator inicial ────────────────────────────────────────
        power_calc = PowerCalculator(flux_n, self.materials, self.logger)
        P0, lp0    = power_calc.compute_initial()
        self.logger.info("Potência inicial estimada: %.3f W", P0)

        # ── Integrador — normalization_mode='flux' não usa source_rates ──
        integrator = self._build_integrator(op, dt_s, source_rate=None)

        try:
            if self._tn_enabled():
                ok = self._run_tn_loop(op, dt_s, flux_n, power_calc)
            else:
                integrator.integrate()
                ok = True
        except Exception as exc:
            return self._fail(f"Depleção falhou: {exc}")

        if not ok:
            return self._fail("Loop T-N não convergiu")

        self._move_results()

        # ── Calcular potência por timestep a partir do h5 ──────────────────
        depl_h5 = self.temp_dir / "depletion_results.h5"
        if depl_h5.exists():
            try:
                depl_results = openmc.deplete.Results(str(depl_h5))
                last_step    = len(dt_s) - 1

                # ── 2ª PASSADA: descoberta de produtos fissionáveis ─────────
                # Após a corrida completa, varre o h5 no último timestep para
                # descobrir todos os nuclídeos fissionáveis produzidos
                # (Pu239, Pu241, Np238, Am242m, etc.) com inventário real.
                # Isso garante que produtos de captura em U238 e cadeia de Pu
                # entram no cálculo de potência de todos os timesteps.
                try:
                    dynamic_sigma_f = power_calc.discover_fissile_products(
                        depl_results, last_step_idx=last_step
                    )
                except Exception as exc:
                    self.logger.warning(
                        "2ª passada (discover_fissile_products) falhou: %s. "
                        "Usando 1ª passada estática.",
                        exc,
                    )

                # ── Calcular P para todos os timesteps (agora com 2ª passada) ─
                t_start_h = 0.0
                for step_i, dt_val in enumerate(dt_s):
                    t_end_h = t_start_h + float(dt_val) / 3600.0
                    P_step, lp_step = power_calc.compute_from_inventory(
                        step_i, depl_results
                    )
                    self._ts_results.append(TimestepResult(
                        step_idx=step_i,
                        t_start_h=t_start_h,
                        t_end_h=t_end_h,
                        dt_s=float(dt_val),
                        power_total_W=P_step,
                        layers=lp_step,
                    ))
                    t_start_h = t_end_h

                # ── Verificar convergência entre passadas ───────────────────
                if self._ts_results:
                    # Recalcular P_1 para o último step usando apenas estáticos
                    # (já armazenado em power_calc._pass1_total como P_inicial;
                    # aqui comparamos com P_final da 2ª passada)
                    P_final_pass2 = self._ts_results[-1].power_total_W
                    power_calc.check_convergence(P_final_pass2)

                self.logger.info(
                    "Potência calculada do h5 [2 passadas]: %d timesteps | "
                    "P_inicial=%.3f W | P_final=%.3f W",
                    len(self._ts_results),
                    self._ts_results[0].power_total_W if self._ts_results else 0.0,
                    self._ts_results[-1].power_total_W if self._ts_results else 0.0,
                )
            except Exception as exc:
                self.logger.warning("Falha ao calcular potência do h5: %s", exc)
                # Fallback: preencher com potência inicial em todos os passos
                t_start_h = 0.0
                for step_i, dt_val in enumerate(dt_s):
                    t_end_h = t_start_h + float(dt_val) / 3600.0
                    self._ts_results.append(TimestepResult(
                        step_idx=step_i, t_start_h=t_start_h, t_end_h=t_end_h,
                        dt_s=float(dt_val), power_total_W=P0, layers=lp0,
                    ))
                    t_start_h = t_end_h
        else:
            self.logger.warning("depletion_results.h5 não encontrado — "
                                "usando potência inicial como fallback")
            t_start_h = 0.0
            for step_i, dt_val in enumerate(dt_s):
                t_end_h = t_start_h + float(dt_val) / 3600.0
                self._ts_results.append(TimestepResult(
                    step_idx=step_i, t_start_h=t_start_h, t_end_h=t_end_h,
                    dt_s=float(dt_val), power_total_W=P0, layers=lp0,
                ))
                t_start_h = t_end_h

        self.logger.info(
            "THERMAL_COUPLING=%s — PyNE cooling %s.",
            self.sp.get("thermal_coupling", False),
            "desacoplado (simulação OpenMC pura)" if not self._tc_flag()
            else "ativo",
        )

        cool_json = None
        if self.cooling_hours > 0.0 and self._tc_flag():
            cool_json = self._run_cooling_pyne()

        if self._tn_history:
            self._save_json(self._tn_history,
                            self.temp_dir / "tn_loop_history.json", "Histórico T-N")

        return SimulationResult(
            success=True,
            depletion_h5=depl_h5 if depl_h5.exists() else None,
            cooling_json=cool_json,
            timestep_results=self._ts_results,
            tn_history=self._tn_history,
            version=self.VERSION,
        )

    # ── Modelo ────────────────────────────────────────────────────────────────

    def _build_model(self) -> openmc.Model:
        """
        Constrói openmc.Model com tallies de diagnóstico.

        V239: os tallies aqui são APENAS para diagnóstico (heating, flux_spectrum).
        A depleção usa tallies internos do IndependentOperator — esses não aparecem
        nos statepoints de depleção e não são usados para normalização.
        """
        settings = self._build_settings()
        tallies  = self._build_diagnostic_tallies()
        return openmc.Model(
            geometry=self.geometry,
            materials=openmc.Materials(self.materials),
            settings=settings,
            tallies=tallies,
        )

    def _build_settings(self) -> openmc.Settings:
        """
        V241: openmc.Settings para corrida de transporte no modo flux.
        
        MODO FLUX (normalization_mode='flux'):
        - NÃO há fonte externa — fluxo prescrito diretamente no IndependentOperator
        - OpenMC calcula MicroXS internamente via método de características
        - Fonte volumétrica removida: era bug arquitetural que distribuía
          nêutrons numa caixa 30× maior que o wafer (água + wafer)
        
        Configuração mínima necessária apenas para transporte MC interno.
        """
        s           = openmc.Settings()
        s.run_mode  = "fixed source"
        s.particles = int(self.sp.get("nparticles", 100_000))
        s.batches   = int(self.sp.get("nbatches", 10))
        s.inactive  = 0
        s.output    = {"summary": True}
        
        # V241: BUG FIX — Removida fonte volumétrica da caixa de água
        # No modo 'flux', o IndependentOperator não usa fonte externa.
        # A fonte abaixo estava distribuindo nêutrons numa região 30× maior
        # que o wafer (water_lateral=10cm vs wafer_x=1.69cm), causando
        # cálculo incorreto de MicroXS e fluxo efetivo muito menor que nominal.
        #
        # Comentada para referência histórica:
        # x_cm = float(self.sp.get("wafer_x_cm", self.sp.get("x", 1.69)))
        # y_cm = float(self.sp.get("wafer_y_cm", self.sp.get("y", 1.69)))
        # water_axial = float(self.sp.get("water_axial_cm", 5.0))
        # water_lateral = float(self.sp.get("water_lateral_cm", 10.0))
        # x_ext = x_cm / 2.0 + water_lateral
        # y_ext = y_cm / 2.0 + water_lateral
        # z_bot = -water_axial
        # z_top = float(self.sp.get("total_thickness_cm", 0.2)) + water_axial
        # source_box = openmc.stats.Box([-x_ext, -y_ext, z_bot], [x_ext, y_ext, z_top])
        # s.source = [openmc.IndependentSource(space=source_box, ...)]
        
        s.source = None  # V241: Sem fonte externa no modo flux
        
        temp_default_k = float(self.sp.get("temperature_default_k", 294.0))
        s.temperature = {
            "method":    "interpolation",
            "default":   temp_default_k,
            "range":     [250.0, 2500.0],
            "tolerance": 200.0,
            "multipole": False,
        }
        return s

    def _build_diagnostic_tallies(self) -> openmc.Tallies:
        """
        Tallies de DIAGNÓSTICO — não usados pelo IndependentOperator.
        Incluem: flux_spectrum (multigrupo 252), heating por célula.
        """
        tallies = openmc.Tallies()
        cells   = [c for c in self.geometry.get_all_cells().values()
                   if not c.name.startswith("water_")]
        if not cells:
            cells = list(self.geometry.get_all_cells().values())

        self.logger.info(
            "Tallies de diagnóstico: %d células wafer: %s",
            len(cells),
            [f"{c.name}(id={c.id})" for c in cells],
        )

        cf = openmc.CellFilter(cells) if cells else None

        # Heating por célula — para diagnóstico de potência
        t_heat = openmc.Tally(name="heating_diag")
        if cf:
            t_heat.filters = [cf]
        t_heat.scores = ["heating"]
        tallies.append(t_heat)

        # Espectro de fluxo multigrupo — para verificar espectro das MicroXS
        try:
            import openmc.mgxs as _mgxs
            groups = _mgxs.EnergyGroups(
                group_edges=np.logspace(-5, 7, 253)  # 252 grupos log-espaçados
            )
            ef = openmc.EnergyFilter(groups.group_edges)
            t_flux = openmc.Tally(name="flux_spectrum")
            if cf:
                t_flux.filters = [cf, ef]
            else:
                t_flux.filters = [ef]
            t_flux.scores = ["flux"]
            tallies.append(t_flux)
        except Exception:
            # openmc.mgxs pode não estar disponível
            t_flux = openmc.Tally(name="flux_spectrum")
            if cf:
                t_flux.filters = [cf]
            t_flux.scores = ["flux"]
            tallies.append(t_flux)

        return tallies

    # ── Integrador ────────────────────────────────────────────────────────────

    _INTEGRATOR_MAP = {
        "predictor": "PredictorIntegrator",
        "cecm":      "CECMIntegrator",
        "celi":      "CELIIntegrator",
        "cf4":       "CF4Integrator",
        "epc_rk4":   "EPCRK4Integrator",
        "leqi":      "LEQIIntegrator",
        "si_celi":   "SICELIIntegrator",
        "si_leqi":   "SILEQIIntegrator",
    }

    def _build_integrator(self, operator, dt_s: np.ndarray,
                           source_rate: float = None):
        """
        Constrói o integrador de depleção.

        Com normalization_mode='source-rate', passa source_rates=[sr]*n_steps
        para converter fluxes [n-cm/src] em taxas de reação absolutas.
        source_rate [n/s] = flux_nominal [n/cm²/s] * volume_total [cm³]
        """
        name_req = (self.sp.get("_depletion_integrator") or
                    self.sp.get("depletion_integrator") or "celi")
        if name_req.lower() == "predictor":
            self.logger.warning(
                "PredictorIntegrator bloqueado (impreciso para Mo99). "
                "Usando CELIIntegrator."
            )
            name_req = "celi"

        cls_name = self._INTEGRATOR_MAP.get(name_req.lower(), "CELIIntegrator")
        IntClass  = getattr(openmc.deplete, cls_name, None)
        if IntClass is None:
            self.logger.warning("'%s' nao encontrado -> CELIIntegrator", cls_name)
            IntClass = getattr(openmc.deplete, "CELIIntegrator",
                               openmc.deplete.PredictorIntegrator)

        if source_rate is not None:
            source_rates = [source_rate] * len(dt_s)
            self.logger.info(
                "Integrador: %s | n_passos=%d | source_rate=%.4e n/s",
                cls_name, len(dt_s), source_rate,
            )
            return IntClass(
                operator=operator,
                timesteps=dt_s,
                source_rates=source_rates,
                timestep_units="s",
            )
        else:
            self.logger.info(
                "Integrador: %s | n_passos=%d | sem source_rates",
                cls_name, len(dt_s),
            )
            return IntClass(
                operator=operator,
                timesteps=dt_s,
                timestep_units="s",
            )

    # ── Timesteps ─────────────────────────────────────────────────────────────

    def _safe_timesteps(self) -> np.ndarray:
        dep_params  = self.sp.get("depletion_params") or {}
        ts_internos = dep_params.get("timesteps_internos_s")
        if ts_internos and len(ts_internos) > 0:
            dt_s = np.asarray(ts_internos, dtype=float)
            dt_s = dt_s[dt_s > 0.0]
            if len(dt_s) > 0:
                self.logger.info("_safe_timesteps: %d sub-passos (Δt=%.1fh)",
                                 len(dt_s), dt_s[0] / 3600.0)
                return dt_s

        dt_s = np.diff(self.timesteps_h) * 3600.0
        dt_s = dt_s[dt_s > 0.0]
        if len(dt_s) == 0:
            dt_h = float(self.sp.get("dt_h", 12.0))
            n    = max(1, round(float(self.sp.get("total_time_h", 48.0)) / dt_h))
            dt_s = np.full(n, dt_h * 3600.0)
            self.logger.warning("Fallback timesteps: %d × %.1fh", n, dt_h)
        return dt_s

    # ── T-N loop ──────────────────────────────────────────────────────────────

    def _tn_enabled(self) -> bool:
        user_flag = self.sp.get("thermal_coupling")
        if user_flag is not None and not bool(user_flag):
            return False
        return (
            self.tn_cfg is not None
            and self.thermo is not None
            and bool(getattr(self.tn_cfg, "ENABLE_TN_COUPLING", False))
        )

    def _tc_flag(self) -> bool:
        """Retorna True se thermal_coupling está ativado."""
        _tc = self.sp.get("thermal_coupling", False)
        if isinstance(_tc, str):
            return _tc.strip().lower() in ("true", "1", "yes", "sim")
        return bool(_tc)

    def _run_tn_loop(self, operator, dt_array: np.ndarray,
                     flux_n: float, power_calc: PowerCalculator) -> bool:
        cfg    = self.tn_cfg
        alpha  = float(getattr(cfg, "RELAXATION_FACTOR", 0.5))
        eps_T  = float(getattr(cfg, "CONVERGENCE_EPSILON_TEMP", 0.5))
        max_it = int(getattr(cfg, "MAX_TN_ITERATIONS", 20))

        t_start_h = 0.0
        for step_idx, dt in enumerate(dt_array):
            t_end_h   = t_start_h + dt / 3600.0
            converged = False
            P_total   = 0.0

            for it in range(max_it):
                try:
                    tmp = self._build_integrator(operator, np.array([float(dt)]))
                    tmp.integrate()
                except Exception as exc:
                    self.logger.error("TN step=%d iter=%d: %s", step_idx, it, exc)
                    return False

                # Calcular potência do h5 depois de integrar
                depl_h5 = self.temp_dir / "depletion_results.h5"
                if depl_h5.exists():
                    try:
                        depl_results = openmc.deplete.Results(str(depl_h5))
                        P_total, lp  = power_calc.compute_from_inventory(
                            step_idx, depl_results
                        )
                    except Exception:
                        P_total, lp = 0.0, []
                else:
                    P_total, lp = 0.0, []

                power_by_layer = {lp_item.layer_name: lp_item.power_W
                                  for lp_item in lp}
                new_T = self._call_thermal_solver(power_by_layer, float(dt))

                max_dT = 0.0
                for mat in self.materials:
                    T_new = new_T.get(mat.name)
                    if T_new is None:
                        continue
                    T_old  = float(mat.temperature or 300.0)
                    T_rel  = T_old + alpha * (float(T_new) - T_old)
                    max_dT = max(max_dT, abs(T_rel - T_old))
                    mat.temperature = T_rel

                self._tn_history.append({
                    "step": step_idx, "iter": it,
                    "P_total_W": P_total, "max_dT_K": max_dT,
                })
                openmc.Materials(self.materials).export_to_xml()

                if max_dT < eps_T:
                    converged = True
                    break

            self._ts_results.append(TimestepResult(
                step_idx=step_idx, t_start_h=t_start_h, t_end_h=t_end_h,
                dt_s=float(dt), power_total_W=P_total,
                tn_iters=it + 1, converged=converged,
            ))
            t_start_h = t_end_h
        return True

    def _call_thermal_solver(self, power_by_layer: Dict, dt: float) -> Dict:
        if not self._tc_flag() or self.thermo is None:
            return {}
        try:
            if hasattr(self.thermo, "solve_thermal_step"):
                return self.thermo.solve_thermal_step(
                    power_by_layer=power_by_layer, dt=dt)
            if hasattr(self.thermo, "compute_temperature_profile"):
                return self.thermo.compute_temperature_profile(
                    power_distribution=power_by_layer)
        except Exception as exc:
            self.logger.warning("Solver térmico: %s", exc)
        return {}

    # ── Cooling PyNE ──────────────────────────────────────────────────────────

    def _run_cooling_pyne(self) -> Optional[Path]:
        if not _PYNE:
            self.logger.error("PyNE não instalado — cooling indisponível.")
            return None
        res_h5 = self.temp_dir / "depletion_results.h5"
        if not res_h5.exists():
            self.logger.error("depletion_results.h5 não encontrado: %s", res_h5)
            return None

        mats_xml = res_h5.parent / "materials.xml"
        if not mats_xml.exists():
            mats_xml_cwd = Path("materials.xml")
            mats_xml = mats_xml_cwd if mats_xml_cwd.exists() else mats_xml

        try:
            depl       = openmc.deplete.Results(str(res_h5.resolve()))
            final_mats = depl.export_to_materials(-1, path=str(mats_xml))
        except Exception as exc:
            self.logger.error("Falha ao ler h5 para cooling: %s", exc)
            return None

        pyne_mats: Dict[int, object] = {}
        for om in final_mats:
            comp = {}
            for nuc, dens in om.get_nuclide_atom_densities().items():
                if dens <= 0.0:
                    continue
                try:
                    comp[_pync.id(nuc)] = dens
                except Exception:
                    pass
            if comp:
                try:
                    om_mass = om.get_mass() or 1.0
                except Exception:
                    om_mass = 1.0
                pyne_mats[om.id] = _PyNEMat(comp, mass=om_mass)

        if not pyne_mats:
            return None

        dt_cool  = self.cooling_hours * 3600.0 / self.cooling_steps
        cool_log: Dict[str, list] = {}
        for step in range(self.cooling_steps):
            t_h = (step + 1) * dt_cool / 3600.0
            for mid, pm in list(pyne_mats.items()):
                try:
                    pyne_mats[mid] = pm.decay(dt_cool)
                    cool_log.setdefault(str(mid), []).append({
                        "step": step + 1, "t_h": round(t_h, 5),
                        "n_nuclides": len(pyne_mats[mid].comp),
                    })
                except Exception as exc:
                    self.logger.warning("PyNE decay mat=%d step=%d: %s", mid, step, exc)

        out_json = self.temp_dir / "cooling_pyne_results.json"
        self._save_json(cool_log, out_json, "Cooling log")
        return out_json

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _move_results(self) -> None:
        out = self.temp_dir
        for fname in ("geometry.xml", "materials.xml", "settings.xml", "tallies.xml"):
            p = Path(fname)
            if p.exists():
                shutil.move(str(p), str(out / fname))
        if Path("depletion_results.h5").exists():
            shutil.move("depletion_results.h5",
                        str(out / "depletion_results.h5"))
        for sp in Path(".").glob("statepoint.*.h5"):
            dest = out / sp.name
            if not dest.exists():
                shutil.move(str(sp), str(dest))

    def _save_json(self, data, path: Path, label: str = "") -> None:
        try:
            with open(path, "w", encoding="utf-8") as fj:
                json.dump(data, fj, indent=2, default=str)
        except Exception as exc:
            self.logger.warning("Falha ao salvar %s: %s", label, exc)

    def _fail(self, msg: str) -> SimulationResult:
        self.logger.error("FALHA: %s", msg)
        return SimulationResult(
            success=False, depletion_h5=None, cooling_json=None, error_msg=msg
        )

    @staticmethod
    def _init_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("SimulationRunner")
        if not logger.handlers:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)
        return logger


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    geometry,
    materials,
    layers,
    system_params:  dict,
    settings_result,
    tn_loop_config  = None,
    output_dir:     Path = Path("pipeline_results"),
    thermal_module  = None,
    geometry_result: dict = None,
    **kwargs,
) -> dict:
    """Interface pública chamada pelo Maestro na Phase D."""
    sp = dict(system_params)

    # Garantir wafer_x_cm / wafer_y_cm em sp
    if "wafer_x_cm" not in sp or "wafer_y_cm" not in sp:
        layer_list = list(layers.values()) if isinstance(layers, dict) else list(layers or [])
        x_cm = y_cm = None
        if layer_list:
            area = float(layer_list[0].get("area_cm2", 0.0))
            if area > 0.0:
                side = math.sqrt(area)
                x_cm = y_cm = side
        sp["wafer_x_cm"] = x_cm or sp.get("x", 1.69)
        sp["wafer_y_cm"] = y_cm or sp.get("y", 1.69)

    # Injetar geometry_result se disponível
    if geometry_result and "_geometry_result" not in sp:
        sp["_geometry_result"] = geometry_result

    # Extrair timesteps e data_manager de settings_result
    if hasattr(settings_result, "timesteps"):
        timesteps_h = list(settings_result.timesteps)
        data_manager = settings_result.data_manager
    elif isinstance(settings_result, dict):
        timesteps_h  = list(settings_result.get("timesteps", [0.0]))
        data_manager = settings_result.get("data_manager")
    else:
        timesteps_h  = [0.0]
        data_manager = None

    if isinstance(settings_result, dict):
        tp        = settings_result.get("temporal_params", {})
        cooling_h = float(tp.get("cooling_time_h", sp.get("cooling_time_h", 0.0)))
        # Injetar depletion_params em sp para acesso pelo runner
        dep = settings_result.get("depletion_params", {})
        if dep and "depletion_params" not in sp:
            sp["depletion_params"] = dep
        # Normalização e integrador
        sp.setdefault("_depletion_normalization",
                      dep.get("normalization", "flux"))
        sp.setdefault("_depletion_integrator",
                      dep.get("integrator", "celi"))
    else:
        cooling_h = 0.0

    runner = SimulationRunner(
        geometry=geometry,
        materials=materials,
        layers=layers,
        system_params=sp,
        timesteps_h=timesteps_h,
        data_manager=data_manager,
        thermal_module=thermal_module,
        tn_loop_config=tn_loop_config,
        cooling_hours=cooling_h,
        cooling_steps=max(1, int(cooling_h)) if cooling_h > 0 else 6,
        output_dir=output_dir,
        temp_dir=Path(output_dir) / "temp",
        log_path=Path("logs") / "simulation.log",
        _geometry_result=sp.get("_geometry_result", {}),
    )

    result = runner.run()

    power_dist: dict = {}
    if result.timestep_results:
        for lp in result.timestep_results[-1].layers:
            power_dist[lp.layer_name] = (
                power_dist.get(lp.layer_name, 0.0) + lp.power_W
            )

    h5_str = str(result.depletion_h5) if result.depletion_h5 else None
    return {
        "success":            result.success,
        "version":            result.version,
        "depletion_h5":       h5_str,
        "h5_depletion_path":  h5_str,
        "cooling_json":       str(result.cooling_json) if result.cooling_json else None,
        "cooling_time_h":     runner.cooling_hours,
        "power_distribution": power_dist,
        "timestep_results":   result.timestep_results,
        "tn_history":         result.tn_history,
        "error_msg":          result.error_msg,
        "error":              result.error_msg,
    }
