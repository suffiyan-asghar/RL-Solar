"""One air-conditioned room as a first-order RC model.

    C dT/dt = (T_out - T) / R + A * G / 1000 + Q_int - Q_ac

T room temperature (C), T_out outdoor air (C), G global horizontal irradiance (W/m2),
R thermal resistance (C/kW), C capacitance (kWh/C), A effective solar aperture (m2),
Q_int internal gains (kW), Q_ac cooling delivered by the AC (kW thermal) = COP x P_elec.

With constant inputs over the hour the solution is exact:
    T(t) = T_eq + (T0 - T_eq) exp(-t / RC),   T_eq = T_out + R (A G / 1000 + Q_int - Q_ac)

The AC is a thermostat: it delivers the cooling needed to reach the setpoint by the end of
the hour, limited by its capacity. Comfort = degree-hours of the hour-mean temperature above
the normal setpoint during occupied hours.
"""
import math


class Room:
    def __init__(self, cc, initial_temp=None):
        self.cc = cc
        self.tau = cc.r_c_per_kw * cc.c_kwh_per_c          # hours
        self.decay = math.exp(-1.0 / self.tau)
        self.temp = cc.initial_temp_c if initial_temp is None else initial_temp

    def _t_eq(self, t_out, ghi, q_cool, cc=None):
        cc = cc or self.cc
        return t_out + cc.r_c_per_kw * (cc.solar_aperture_m2 * ghi / 1000.0 + cc.internal_gain_kw - q_cool)

    def cooling_needed(self, t_out, ghi, setpoint, cc=None):
        """Thermal kW needed to reach `setpoint` at the end of this hour (0..capacity)."""
        cc = cc or self.cc
        t_eq_target = (setpoint - self.temp * self.decay) / (1.0 - self.decay)
        q = (self._t_eq(t_out, ghi, 0.0, cc) - t_eq_target) / cc.r_c_per_kw
        return min(max(q, 0.0), cc.ac_cooling_kw)

    def step(self, t_out, ghi, q_cool, cc=None):
        """Advance one hour with `q_cool` kW of cooling. Returns (mean temp, end temp)."""
        t_eq = self._t_eq(t_out, ghi, q_cool, cc)
        t0 = self.temp
        end = t_eq + (t0 - t_eq) * self.decay
        mean = t_eq + (t0 - t_eq) * self.tau * (1.0 - self.decay)
        self.temp = end
        return mean, end
