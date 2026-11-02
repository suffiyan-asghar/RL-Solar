"""Electricity bill for one hour of energy flows (Rs).

    import charge = imported kWh x time-of-use rate x (1 + energy taxes)
    fixed charge  = monthly fixed charge x (1 + fixed taxes), spread evenly over the month's hours
    export credit = exported kWh x export rate (net billing buyback, or the legacy
                    net-metering rate as a scenario)

Rates live in configs/tariffs/*.yaml. Simplifications: no slab (block) rates, no
billing-period netting of units under net metering (exports are credited at a fixed rate),
taxes do not apply to export credits.
"""
import calendar
import warnings

import numpy as np


class Tariff:
    def __init__(self, tc):
        self.tc = tc
        rates = np.full((12, 24), tc.import_.default_rate, dtype=float)
        for p in tc.import_.periods:
            for m in p.months:
                rates[m - 1, p.hours] = p.rate
        self.rates = rates
        self.energy_tax = sum(t.percent for t in tc.taxes if "energy" in t.applies_to) / 100.0
        self.fixed_tax = sum(t.percent for t in tc.taxes if "fixed" in t.applies_to) / 100.0
        self.export_rate = tc.export.rate
        if tc.verified is False:
            warnings.warn(f"tariff '{tc.name}' uses UNVERIFIED placeholder values", stacklevel=2)

    def import_rate(self, month, hour):
        """Pre-tax import rate (Rs/kWh)."""
        return self.rates[month - 1, hour]

    def hour_cost(self, imp, exp, month, hour, year=2023):
        rate = self.rates[month - 1, hour]
        import_cost = imp * rate
        if self.energy_tax:
            import_cost = import_cost * (1.0 + self.energy_tax)
        export_credit = exp * self.export_rate
        fixed = 0.0
        if self.tc.fixed_monthly_rs:
            hours = calendar.monthrange(year, month)[1] * 24
            fixed = self.tc.fixed_monthly_rs * (1.0 + self.fixed_tax) / hours
        return import_cost, export_credit, fixed, rate
