#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py V223 — Construção de geometria OpenMC para wafer multicamadas.

CHANGELOG V223 (Modo Flux — campo de reator):
  - _build_water_material(): mat_water.volume agora é definido explicitamente.
    Necessário para que IndependentOperator possa calcular fluxo físico
    [n/cm²/s] = tally [n.cm/src] * source_rate / volume em qualquer módulo
    que consulte o volume da água.
  - A geometria com água é MANTIDA (não simplificada). A água ao redor do
    wafer é necessária para que o transporte MC de cálculo de MicroXS veja
    o espectro moderado correto — representa o canal do reator.
  - Boundary condition das faces externas: REFLECTIVE em vez de vacuum,
    para simular o wafer imerso num campo de fluxo quase uniforme de reator.
    Fronteiras reflectivas eliminam o "vazamento" artificial de nêutrons que
    ocorria com vacuum + fonte plana, e a geometria com água moderadora
    permite ao MC calcular o espectro termalizado correto para as MicroXS.
  - water_front, water_back, water_lateral: volumes explicitamente definidos.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmc

from config import ValidationLimits, GeometryLimits

_VL = ValidationLimits()
_GL = GeometryLimits()

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ContractValidator
# ─────────────────────────────────────────────────────────────────────────────

class ContractValidator:
    _REQUIRED = (
        "success", "openmc_geometry", "openmc_materials",
        "materials_dict", "cells_dict", "layers",
        "wafer_geometry", "metadata", "updater",
    )

    @classmethod
    def validate(cls, result: dict) -> Tuple[bool, str]:
        for f in cls._REQUIRED:
            if f not in result:
                return False, f"Campo obrigatório ausente: '{f}'"
        if not isinstance(result.get("success"), bool):
            return False, "'success' deve ser bool"
        if result.get("success"):
            for f, t in (("materials_dict", dict), ("cells_dict", dict),
                         ("layers", list), ("metadata", dict), ("wafer_geometry", dict)):
                if not isinstance(result.get(f), t):
                    return False, f"'{f}' deve ser {t.__name__}"
            if result.get("updater") is None:
                return False, "'updater' é None com success=True"
            if result.get("openmc_materials") is None:
                return False, "'openmc_materials' é None com success=True"
            missing = set(result.get("cells_dict", {})) - set(result.get("materials_dict", {}))
            if missing:
                return False, f"Células sem material em materials_dict: {missing}"
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# SharedSurfaceManager
# ─────────────────────────────────────────────────────────────────────────────

class SharedSurfaceManager:
    def __init__(self) -> None:
        self._cache: Dict[str, openmc.ZPlane] = {}

    def get_or_create(self, z: float, boundary_type: Optional[str] = None) -> openmc.ZPlane:
        key = f"{z:.8f}"
        if key not in self._cache:
            plane = openmc.ZPlane(z0=float(z))
            if boundary_type:
                plane.boundary_type = boundary_type
            self._cache[key] = plane
        elif boundary_type and self._cache[key].boundary_type != boundary_type:
            self._cache[key].boundary_type = boundary_type
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
# GeometryUpdater
# ─────────────────────────────────────────────────────────────────────────────

class GeometryUpdater:
    def __init__(self, materials_dict: Dict[str, openmc.Material],
                 cells_dict: Dict[str, openmc.Cell]) -> None:
        self._materials = materials_dict
        self._cells = cells_dict
        self._update_count = 0
        self._needs_export = False
        self._user_temps: Dict[str, float] = {
            name: getattr(mat, "_temperature_user", mat.temperature or 300.0)
            for name, mat in materials_dict.items()
        }

    def get_user_temperature(self, cell_name: str) -> Optional[float]:
        return self._user_temps.get(cell_name)

    def update_temperatures(self, temp_map: Dict[str, float]) -> None:
        self._update_count += 1
        updated = 0
        for cell_name, t_raw in temp_map.items():
            t_k = float(np.clip(float(t_raw), _VL.TEMP_MIN_K, _VL.TEMP_MAX_K))
            if cell_name in self._materials:
                mat = self._materials[cell_name]
                self._user_temps[cell_name] = t_k
                mat.temperature = GeometryBuilder._snap_temperature(t_k)
                updated += 1
            else:
                _log.warning("'%s' não encontrado em materials_dict", cell_name)
        if updated > 0:
            self._needs_export = True

    def mark_exported(self) -> None:
        self._needs_export = False

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def needs_export(self) -> bool:
        return self._needs_export


# ─────────────────────────────────────────────────────────────────────────────
# GeometryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class GeometryBuilder:
    """Constrói geometria OpenMC para wafer multicamadas em campo de reator."""

    VERSION = "V223"

    WATER_AXIAL_CM:   float = 5.0
    WATER_LATERAL_CM: float = 10.0

    _LIB_TEMPS_K: Tuple[float, ...] = (250.0, 294.0, 600.0, 900.0, 1200.0, 2500.0)

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._surface_mgr = SharedSurfaceManager()
        self._errors: List[str] = []

    def build(self, parser_result: dict) -> dict:
        self._errors = []
        if not parser_result.get("success"):
            return self._fail("parser_result.success=False: " + str(parser_result.get("error", "")))
        try:
            wafer_geom = self._extract_wafer_geometry(parser_result)
            layers = self._normalize_layers(parser_result["layers"])

            z = 0.0
            enriched = []
            for lay in layers:
                thick_cm = self._thick_cm(lay)
                lay_e = dict(lay)
                lay_e["zmin"] = round(z, 9)
                lay_e["zmax"] = round(z + thick_cm, 9)
                lay_e["area_cm2"] = wafer_geom["area_cm2"]
                enriched.append(lay_e)
                z += thick_cm
            layers = enriched

            self._validate_layers(layers)
            self._validate_nanoscale(layers)

            materials_dict, openmc_mats = self._build_materials(layers, wafer_geom)
            cells_dict_all, openmc_geometry = self._build_geometry(
                layers, materials_dict, openmc_mats, wafer_geom
            )

            water_cells = {k: v for k, v in cells_dict_all.items() if k.startswith("water_")}
            wafer_cells = {k: v for k, v in cells_dict_all.items() if not k.startswith("water_")}

            updater = GeometryUpdater(materials_dict, wafer_cells)

            result = {
                "success": len(self._errors) == 0,
                "openmc_geometry": openmc_geometry,
                "openmc_materials": openmc_mats,
                "materials_dict": materials_dict,
                "cells_dict": wafer_cells,
                "cells_dict_all": cells_dict_all,
                "water_cells": water_cells,
                "wafer_geometry": wafer_geom,
                "layers": layers,
                "errors": list(self._errors),
                "water_geometry": {
                    "axial_cm": self.WATER_AXIAL_CM,
                    "lateral_cm": self.WATER_LATERAL_CM,
                    "total_z_cm": wafer_geom["total_thickness_cm"] + 2.0 * self.WATER_AXIAL_CM,
                    "total_x_cm": wafer_geom["x_cm"] + 2.0 * self.WATER_LATERAL_CM,
                    "total_y_cm": wafer_geom["y_cm"] + 2.0 * self.WATER_LATERAL_CM,
                },
                "metadata": {
                    "version": self.VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "n_layers": len(layers),
                    "n_materials_total": len(materials_dict),
                    "n_materials_wafer": len(wafer_cells),
                    "n_cells_wafer": len(wafer_cells),
                    "n_cells_total": len(cells_dict_all),
                    "n_water_cells": len(water_cells),
                    "n_surfaces": len(self._surface_mgr),
                    "area_cm2": wafer_geom["area_cm2"],
                    "total_thickness_cm": wafer_geom["total_thickness_cm"],
                    "water_axial_cm": self.WATER_AXIAL_CM,
                    "water_lateral_cm": self.WATER_LATERAL_CM,
                    "boundary_type": "reflective",  # V223: modo reator
                    "shared_materials_allowed": True,
                },
                "updater": updater,
            }

            ok, msg = ContractValidator.validate(result)
            if not ok:
                return self._fail(msg)
            return result

        except Exception as exc:
            _log.exception("GeometryBuilder.build() — exceção: %s", exc)
            return self._fail(str(exc))

    # ── Normalização ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_layers(raw) -> list:
        if isinstance(raw, dict):
            items = [l for l in raw.values() if isinstance(l, dict)]
        elif isinstance(raw, (list, tuple)):
            items = [l for l in raw if isinstance(l, dict)]
        else:
            return []
        if items and all("number" in l for l in items):
            items = sorted(items, key=lambda x: int(x["number"]))
        return items

    @staticmethod
    def _thick_cm(lay: dict) -> float:
        if lay.get("thickness_cm") is not None:
            return float(lay["thickness_cm"])
        return float(lay.get("thickness_mm", 1.0)) / 10.0

    def _extract_wafer_geometry(self, parser_result: dict) -> dict:
        wg = parser_result.get("wafer_geometry", {})
        x_cm = float(wg.get("x_cm") or wg.get("x") or 1.69)
        y_cm = float(wg.get("y_cm") or wg.get("y") or 1.69)
        water_temp_k = float(wg.get("water_temp_k", 300.0))
        layers = self._normalize_layers(parser_result["layers"])
        total_cm = sum(self._thick_cm(l) for l in layers)
        return {
            "x_cm": x_cm, "y_cm": y_cm,
            "area_cm2": x_cm * y_cm,
            "total_thickness_cm": total_cm,
            "water_temp_k": water_temp_k,
        }

    # ── Validações ────────────────────────────────────────────────────────────

    def _validate_layers(self, layers: list) -> None:
        if not layers:
            raise ValueError("Lista de camadas vazia.")
        for i, lay in enumerate(layers):
            if self._thick_cm(lay) <= 0.0:
                raise ValueError(f"Camada '{self._layer_name(lay, i)}': thickness <= 0")

    def _validate_nanoscale(self, layers: list) -> None:
        for i, lay in enumerate(layers):
            thick_cm = self._thick_cm(lay)
            if thick_cm < _GL.NANOSCALE_MIN_CM:
                _log.warning("NANOSCALE: '%s' esp=%.2e cm < %.2e cm",
                             self._layer_name(lay, i), thick_cm, _GL.NANOSCALE_MIN_CM)

    # ── Materiais ─────────────────────────────────────────────────────────────

    @staticmethod
    def _snap_temperature(t_k: float) -> float:
        return min(GeometryBuilder._LIB_TEMPS_K, key=lambda t: abs(t - t_k))

    def _build_materials(self, layers: list, wafer_geom: dict
                         ) -> Tuple[Dict[str, openmc.Material], openmc.Materials]:
        area_cm2 = wafer_geom["area_cm2"]
        materials_dict: Dict[str, openmc.Material] = {}
        mat_list: List[openmc.Material] = []

        for i, lay in enumerate(layers):
            cell_name = self._layer_name(lay, i)
            thick_cm  = self._thick_cm(lay)
            volume_cm3 = area_cm2 * thick_cm

            if volume_cm3 <= 0.0:
                raise ValueError(f"'{cell_name}': volume_cm3={volume_cm3:.4e} <= 0")

            density = self._density_from_layer(lay, volume_cm3)

            if density <= 0.0:
                up = cell_name.upper()
                if any(t in up for t in ("UAL","UO2","U_ME","FUEL","COMBUST","ALVO","TARGET")):
                    raise ValueError(
                        f"'{cell_name}': densidade = {density:.4e} g/cm³. "
                        "Camada combustível com densidade zero — verifique massas isotópicas."
                    )
                _log.warning("'%s': density=%.4e → usando 1.0 g/cm³", cell_name, density)
                density = 1.0

            if density > _VL.RHO_MAX_GCM3:
                msg = f"DENSIDADE IMPOSSÍVEL '{cell_name}': ρ={density:.2f} g/cm³"
                _log.error(msg)
                self._errors.append(msg)

            t_k = float(np.clip(
                float(lay.get("temperature_k") or lay.get("initial_temp_k") or
                      lay.get("temperature") or 300.0),
                _VL.TEMP_MIN_K, _VL.TEMP_MAX_K,
            ))
            t_snapped = self._snap_temperature(t_k)

            mat = openmc.Material(name=cell_name)
            mat.set_density("g/cm3", density)
            mat.volume = volume_cm3          # ← necessário para IndependentOperator
            mat.depletable = True
            mat.temperature = t_snapped
            mat._temperature_user = t_k

            self._add_nuclides(mat, lay, cell_name)
            materials_dict[cell_name] = mat
            mat_list.append(mat)

        return materials_dict, openmc.Materials(mat_list)

    @staticmethod
    def _density_from_layer(lay: dict, volume_cm3: float) -> float:
        for key in ("density_gcm3", "density_g_cm3", "density", "rho"):
            val = lay.get(key)
            if val is not None and float(val) > 0.0:
                return float(val)
        for key in ("total_mass_g", "mass_total_g", "massa_total_g"):
            val = lay.get(key)
            if val is not None and float(val) > 0.0 and volume_cm3 > 0.0:
                return float(val) / volume_cm3
        nuc_map = lay.get("isotopes") or lay.get("fractions_by_mass") or {}
        if isinstance(nuc_map, dict) and nuc_map:
            total_g = sum(float(v) for v in nuc_map.values())
            if total_g > 0.0 and volume_cm3 > 0.0:
                return total_g / volume_cm3
        return 0.0

    @staticmethod
    def _add_nuclides(mat: openmc.Material, lay: dict, cell_name: str) -> int:
        fbm = lay.get("fractions_by_mass")
        iso = lay.get("isotopes")
        frc = lay.get("fractions")

        if fbm and isinstance(fbm, dict):
            nuc_map = {str(k): float(v) for k, v in fbm.items()}
        elif iso and isinstance(iso, dict):
            nuc_map = {str(k): float(v) for k, v in iso.items()}
        elif iso and isinstance(iso, list) and frc and isinstance(frc, list):
            nuc_map = {str(k): float(v) for k, v in zip(iso, frc)}
        else:
            _log.warning("'%s': nenhum formato de nuclídeos reconhecido", cell_name)
            return 0

        total = sum(nuc_map.values())
        if total <= 0.0:
            return 0

        n_added = 0
        for nuc_raw, val in nuc_map.items():
            frac = val / total
            if frac < 1e-14:
                continue
            try:
                mat.add_nuclide(nuc_raw.replace("-", "").strip(), frac, "wo")
                n_added += 1
            except Exception as exc:
                _log.warning("'%s': add_nuclide('%s') falhou: %s", cell_name, nuc_raw, exc)
        return n_added

    # ── Água ──────────────────────────────────────────────────────────────────

    def _build_water_material(self, wafer_geom: dict,
                               water_volume_cm3: float) -> openmc.Material:
        """
        Cria material de água com volume explicitamente definido.

        O volume é necessário para que IndependentOperator e módulos externos
        possam calcular fluxo físico [n/cm²/s] = tally * source_rate / volume.
        A água é depletable=False mas precisa de volume definido.
        """
        t_real = float(wafer_geom.get("water_temp_k", 300.0))
        t_snap = self._snap_temperature(t_real)

        mat = openmc.Material(name="water_reflector")
        mat.add_nuclide("H1",  2.0 / 3.0, "ao")
        mat.add_nuclide("O16", 1.0 / 3.0, "ao")
        mat.set_density("g/cm3", 0.9982)
        mat.depletable = False
        mat.temperature = t_snap
        mat._temperature_user = t_real
        mat.volume = water_volume_cm3   # V223: volume explícito

        try:
            mat.add_s_alpha_beta("c_H_in_H2O")
            _log.info("water_reflector: S(a,b) 'c_H_in_H2O' habilitado")
        except Exception as exc:
            _log.warning("water_reflector: S(a,b) c_H_in_H2O indisponível: %s", exc)

        return mat

    # ── Geometria ─────────────────────────────────────────────────────────────

    def _build_geometry(self, layers: list, materials_dict: Dict[str, openmc.Material],
                         openmc_mats: openmc.Materials, wafer_geom: dict,
                         ) -> Tuple[Dict[str, openmc.Cell], openmc.Geometry]:
        """
        Layout (modo reator, fronteiras reflectivas):

            [reflective] | water_front | wafer multicamada | water_back | [reflective]
            + water_lateral ao redor do wafer (na faixa axial do wafer)

        Fronteiras reflectivas em todas as faces externas representam o wafer
        imerso num campo de fluxo quase uniforme do canal do reator.
        O MC calcula as MicroXS com o espectro moderado pela água ao redor.
        """
        x_cm  = wafer_geom["x_cm"]
        y_cm  = wafer_geom["y_cm"]
        total_wafer_cm = wafer_geom["total_thickness_cm"]
        dax   = self.WATER_AXIAL_CM
        dlat  = self.WATER_LATERAL_CM

        # Volumes das regiões de água
        total_x = x_cm + 2.0 * dlat
        total_y = y_cm + 2.0 * dlat
        vol_front   = total_x * total_y * dax
        vol_back    = total_x * total_y * dax
        vol_lateral = (total_x * total_y - x_cm * y_cm) * total_wafer_cm

        mat_water = self._build_water_material(wafer_geom, vol_front + vol_back + vol_lateral)

        if "water_reflector" not in materials_dict:
            materials_dict["water_reflector"] = mat_water
            openmc_mats.append(mat_water)

        # V223: boundary_type="reflective" em todas as faces externas
        # Representa o wafer imerso em campo de fluxo de reator
        xmin_ext = openmc.XPlane(x0=-(x_cm / 2.0 + dlat), boundary_type="reflective")
        xmax_ext = openmc.XPlane(x0=+(x_cm / 2.0 + dlat), boundary_type="reflective")
        ymin_ext = openmc.YPlane(y0=-(y_cm / 2.0 + dlat), boundary_type="reflective")
        ymax_ext = openmc.YPlane(y0=+(y_cm / 2.0 + dlat), boundary_type="reflective")
        z_ext_bot = openmc.ZPlane(z0=-dax,                  boundary_type="reflective")
        z_ext_top = openmc.ZPlane(z0=total_wafer_cm + dax,  boundary_type="reflective")

        z_wafer_bot = openmc.ZPlane(z0=0.0)
        z_wafer_top = openmc.ZPlane(z0=total_wafer_cm)
        xmin_waf = openmc.XPlane(x0=-x_cm / 2.0)
        xmax_waf = openmc.XPlane(x0=+x_cm / 2.0)
        ymin_waf = openmc.YPlane(y0=-y_cm / 2.0)
        ymax_waf = openmc.YPlane(y0=+y_cm / 2.0)

        self._surface_mgr.clear()
        cells_dict: Dict[str, openmc.Cell] = {}
        z_current = 0.0

        for i, lay in enumerate(layers):
            cell_name = self._layer_name(lay, i)
            z_next    = z_current + self._thick_cm(lay)
            z_bot = self._surface_mgr.get_or_create(z_current)
            z_top = self._surface_mgr.get_or_create(z_next)
            cell  = openmc.Cell(
                name=cell_name,
                fill=materials_dict[cell_name],
                region=(+xmin_waf & -xmax_waf & +ymin_waf & -ymax_waf & +z_bot & -z_top),
            )
            cells_dict[cell_name] = cell
            z_current = z_next

        delta = abs(z_current - total_wafer_cm)
        if delta > _GL.GAP_TOLERANCE_CM:
            _log.warning("GAPS: empilhado=%.10f cm != esperado=%.10f cm (delta=%.2e)",
                         z_current, total_wafer_cm, delta)

        wafer_xy = +xmin_waf & -xmax_waf & +ymin_waf & -ymax_waf

        # Água: cobre toda a extensão lateral (inclui região do wafer) nas faixas axiais
        water_front = openmc.Cell(
            name="water_front", fill=mat_water,
            region=(+xmin_ext & -xmax_ext & +ymin_ext & -ymax_ext & +z_ext_bot & -z_wafer_bot),
        )
        water_back = openmc.Cell(
            name="water_back", fill=mat_water,
            region=(+xmin_ext & -xmax_ext & +ymin_ext & -ymax_ext & +z_wafer_top & -z_ext_top),
        )
        water_lateral = openmc.Cell(
            name="water_lateral", fill=mat_water,
            region=(+xmin_ext & -xmax_ext & +ymin_ext & -ymax_ext &
                    +z_wafer_bot & -z_wafer_top & ~wafer_xy),
        )

        cells_dict["water_front"]   = water_front
        cells_dict["water_back"]    = water_back
        cells_dict["water_lateral"] = water_lateral

        universe = openmc.Universe(cells=list(cells_dict.values()))
        geometry = openmc.Geometry(universe)

        _log.info(
            "_build_geometry: %d camadas wafer + 3 células de água "
            "(front=%.1fcm, back=%.1fcm, lateral=%.1fcm) | boundary=reflective",
            len(layers), dax, dax, dlat,
        )
        return cells_dict, geometry

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _layer_name(lay: dict, idx: int) -> str:
        return str(lay.get("name") or lay.get("material_name") or
                   lay.get("material") or f"layer_{idx + 1}")

    @staticmethod
    def _fail(msg: str) -> dict:
        return {
            "success": False, "error": msg,
            "openmc_geometry": None, "openmc_materials": None,
            "materials_dict": {}, "cells_dict": {}, "cells_dict_all": {},
            "water_cells": {}, "wafer_geometry": {}, "layers": [],
            "errors": [msg], "metadata": {"version": GeometryBuilder.VERSION},
            "updater": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry(parser_result: dict, debug: bool = False) -> dict:
    return GeometryBuilder(debug=debug).build(parser_result)
